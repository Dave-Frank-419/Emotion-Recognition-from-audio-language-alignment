import os

import audmetric
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval_external import ManifestDataset
from ParaCLAP import AUDIO_ENCODER, CLAP, EMBEDDING_DIM, TEXT_ENCODER

CKPT_PATH = os.path.join("ckpt", "best.pth.tar")
MANIFEST = os.path.join("manifests", "ravdess.csv")
WAV_ROOT = "/data/chi-gpu1/ge96xah/data/RAVDESS"


def to_old_keys(state, model):
    remapped = {}
    for key, value in state.items():
        if ".parametrizations.weight.original0" in key:
            remapped[key.replace(".parametrizations.weight.original0", ".weight_g")] = value
        elif ".parametrizations.weight.original1" in key:
            remapped[key.replace(".parametrizations.weight.original1", ".weight_v")] = value
        else:
            remapped[key] = value
    for key, value in model.state_dict().items():
        if "position_ids" in key and key not in remapped:
            remapped[key] = value.clone()
            print(f"patched missing buffer: {key}", flush=True)
    return remapped


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | transformers {transformers.__version__} | torch {torch.__version__}")

    model = CLAP(AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM)
    state = to_old_keys(torch.load(CKPT_PATH, map_location="cpu")["model"], model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing {len(missing)} | unexpected {len(unexpected)}", flush=True)
    if missing or unexpected:
        print(f"missing: {missing[:5]} | unexpected: {unexpected[:5]}", flush=True)
    model.to(device).eval()

    df = pd.read_csv(MANIFEST)
    candidates = sorted(df["label"].unique())
    print(f"ravdess: {len(df)} utterances | candidates: {candidates}")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    tokens = tokenizer(candidates, padding=True, truncation=True,
                       return_tensors="pt").to(device)
    loader = DataLoader(ManifestDataset(df, WAV_ROOT),
                        batch_size=1, num_workers=4, pin_memory=True)

    targets, predictions = [], []
    with torch.no_grad():
        text_emb = F.normalize(model.text_projection(model.text_branch(tokens)), dim=-1)
        for wav, label in loader:
            audio_emb = F.normalize(model.audio_projection(model.audio_branch(wav.to(device))), dim=-1)
            predictions.append(candidates[int((audio_emb @ text_emb.t()).argmax())])
            targets.append(label[0])

    print(f"UAR {audmetric.unweighted_average_recall(targets, predictions):.4f} | "
          f"ACC {audmetric.accuracy(targets, predictions):.4f}", flush=True)
    print(f"prediction counts: {({c: predictions.count(c) for c in candidates})}", flush=True)
