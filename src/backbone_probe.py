import os

import audmetric
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval_external import ManifestDataset
from ParaCLAP import AUDIO_ENCODER, CLAP, EMBEDDING_DIM, TEXT_ENCODER

CKPT_PATH = os.path.join("ckpt", "best.pth.tar")
HANDOVER_PATH = "/nas/student/DavidFrank/handover/paraclap_llm_best_ep59.pth.tar"
NAS_ROOT = "/nas/student/DavidFrank/handover"
MANIFEST = os.path.join("manifests", "ravdess.csv")
WAV_ROOT = "/data/chi-gpu1/ge96xah/data/RAVDESS"


def load_state(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def check_export_integrity(state):
    handover = torch.load(HANDOVER_PATH, map_location="cpu")
    keys = sorted(set(state) | set(handover))
    diff = [k for k in keys
            if k not in state or k not in handover or not torch.equal(state[k], handover[k])]
    print(f"ckpt/best vs handover: {len(state)} vs {len(handover)} tensors | "
          f"{len(diff)} differences", flush=True)
    if diff:
        print(f"first differences: {diff[:5]}", flush=True)


def check_trained_heads(state):
    for prefix in ("audio_projection", "text_projection"):
        w = state[f"{prefix}.layer_norm.weight"]
        b = state[f"{prefix}.layer_norm.bias"]
        lin = state[f"{prefix}.linear1.weight"]
        print(f"{prefix}: layer_norm |w-1| mean {(w - 1).abs().mean():.4f} | "
              f"|b| mean {b.abs().mean():.4f} | linear1 std {lin.std():.4f}", flush=True)
    print(f"logit_scale {float(state['logit_scale']):.4f}", flush=True)


def nas_weight_files(root):
    hits = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith((".pth", ".pt", ".pth.tar", ".ckpt", ".bin")):
                hits.append(os.path.join(dirpath, f))
    return hits


def run_probe(state, loader, candidates, tokenizer, device):
    model = CLAP(AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing {len(missing)} | unexpected {len(unexpected)}", flush=True)
    if missing:
        print(f"missing keys: {missing[:6]}", flush=True)
    model.to(device).eval()

    tokens = tokenizer(candidates, padding=True, truncation=True,
                       return_tensors="pt").to(device)
    targets, predictions = [], []
    with torch.no_grad():
        text_emb = F.normalize(model.text_projection(model.text_branch(tokens)), dim=-1)
        for wav, label in loader:
            audio_emb = F.normalize(model.audio_projection(model.audio_branch(wav.to(device))), dim=-1)
            predictions.append(candidates[int((audio_emb @ text_emb.t()).argmax())])
            targets.append(label[0])

    uar = audmetric.unweighted_average_recall(targets, predictions)
    counts = {c: predictions.count(c) for c in candidates}
    print(f"UAR {uar:.4f} | ACC {audmetric.accuracy(targets, predictions):.4f}", flush=True)
    print(f"prediction counts: {counts}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    state = load_state(CKPT_PATH)
    print("=== export/rename integrity ===", flush=True)
    check_export_integrity(state)
    check_trained_heads(state)

    df = pd.read_csv(MANIFEST)
    candidates = sorted(df["label"].unique())
    print(f"ravdess: {len(df)} utterances | candidates: {candidates}")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    loader = DataLoader(ManifestDataset(df, WAV_ROOT),
                        batch_size=1, num_workers=4, pin_memory=True)

    print("=== control: ckpt/best.pth.tar (epoch 59) ===", flush=True)
    run_probe(state, loader, candidates, tokenizer, device)

    for path in nas_weight_files(NAS_ROOT):
        print(f"=== NAS file: {path} ===", flush=True)
        try:
            run_probe(load_state(path), loader, candidates, tokenizer, device)
        except Exception as error:
            print(f"not loadable as model: {error}", flush=True)

    for name in ("last.pth.tar", "best_loss_ep53.pth.tar"):
        print(f"=== cluster file: ckpt/{name} ===", flush=True)
        run_probe(load_state(os.path.join("ckpt", name)), loader, candidates, tokenizer, device)

    partial_drops = {
        "missing proj layer_norm": ("audio_projection.layer_norm", "text_projection.layer_norm"),
        "missing audio_projection": ("audio_projection",),
        "missing text_projection": ("text_projection",),
    }
    for name, prefixes in partial_drops.items():
        partial = {k: v for k, v in state.items() if not k.startswith(prefixes)}
        print(f"=== simulation: {name} ===", flush=True)
        run_probe(partial, loader, candidates, tokenizer, device)
