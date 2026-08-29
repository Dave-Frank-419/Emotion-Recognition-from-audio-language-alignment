import os

import audmetric
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval_external import ManifestDataset
from ParaCLAP import CLAP, EMBEDDING_DIM, TEXT_ENCODER

CKPT_PATH = os.path.join("ckpt", "best.pth.tar")
HANDOVER_PATH = "/nas/student/DavidFrank/handover/paraclap_llm_best_ep59.pth.tar"
MANIFEST = os.path.join("manifests", "ravdess.csv")
WAV_ROOT = "/data/chi-gpu1/ge96xah/data/RAVDESS"
BACKBONES = [
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "facebook/wav2vec2-large-robust",
    "audeering/wav2vec2-large-robust-24-ft-age-gender",
]


def run_probe(backbone, state, loader, candidates, tokenizer, device, half=False):
    model = CLAP(backbone, TEXT_ENCODER, EMBEDDING_DIM)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"--- {backbone}{' | fp16' if half else ''} ---")
    print(f"missing {len(missing)} | unexpected {len(unexpected)}", flush=True)
    model.to(device).eval()
    if half:
        model.half()

    tokens = tokenizer(candidates, padding=True, truncation=True,
                       return_tensors="pt").to(device)
    targets, predictions = [], []
    with torch.no_grad():
        text_emb = F.normalize(model.text_projection(model.text_branch(tokens)), dim=-1)
        for wav, label in loader:
            x = wav.to(device)
            if half:
                x = x.half()
            audio_emb = F.normalize(model.audio_projection(model.audio_branch(x)), dim=-1)
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

    state = torch.load(CKPT_PATH, map_location="cpu")["model"]
    df = pd.read_csv(MANIFEST)
    candidates = sorted(df["label"].unique())
    print(f"ravdess: {len(df)} utterances | candidates: {candidates}")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    loader = DataLoader(ManifestDataset(df, WAV_ROOT),
                        batch_size=1, num_workers=4, pin_memory=True)

    for backbone in BACKBONES:
        run_probe(backbone, state, loader, candidates, tokenizer, device)
    run_probe(BACKBONES[0], state, loader, candidates, tokenizer, device, half=True)

    print("=== handover file ===", flush=True)
    handover_state = torch.load(HANDOVER_PATH, map_location="cpu")
    run_probe(BACKBONES[0], handover_state, loader, candidates, tokenizer, device)
