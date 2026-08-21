import os

import audmetric
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ParaCLAP import AUDIO_ENCODER, CKPT_DIR, CLAP, EMBEDDING_DIM, SAMPLE_RATE, TEXT_ENCODER

RESULTS_DIR = "results"
SENTENCE_TEMPLATE = "this person is feeling {}"

DATASETS = {
    "ravdess": {"wav_root": "/data/chi-gpu1/ge96xah/data/RAVDESS", "manifest": "manifests/ravdess.csv"},
    "tess": {"wav_root": "/data/chi-gpu1/ge96xah/data/TESS", "manifest": "manifests/tess.csv"},
}


class ManifestDataset(Dataset):

    def __init__(self, df, wav_root):
        self.files = list(df["file"])
        self.labels = list(df["label"])
        self.wav_root = wav_root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            wav, sr = torchaudio.load(os.path.join(self.wav_root, self.files[index]))
        except Exception:
            print(f"audio load failed, using silence: {self.files[index]}", flush=True)
            return torch.zeros(SAMPLE_RATE), self.labels[index]
        if sr != SAMPLE_RATE:
            wav = F_audio.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
        return wav.mean(0), self.labels[index]


def load_model(ckpt_path, device):
    model = CLAP(AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']})")
    return model.to(device).eval()


def evaluate_zero_shot(name, wav_root, manifest, model, tokenizer, device,
                       sentence_template=SENTENCE_TEMPLATE):
    df = pd.read_csv(manifest)
    candidates = sorted(df["label"].unique())
    print(f"{name}: {len(df)} utterances | candidates: {candidates}")

    queries = {
        "word": candidates,
        "sentence": [sentence_template.format(c) for c in candidates],
    }
    text_embs = {}
    with torch.no_grad():
        for style, texts in queries.items():
            tokens = tokenizer(texts, padding=True, truncation=True,
                               return_tensors="pt").to(device)
            text_embs[style] = F.normalize(model.text_projection(model.text_branch(tokens)), dim=-1)

    loader = DataLoader(ManifestDataset(df, wav_root),
                        batch_size=1, num_workers=4, pin_memory=True)
    targets = []
    predictions = {style: [] for style in queries}
    for i, (wav, label) in enumerate(loader):
        with torch.no_grad():
            audio_emb = F.normalize(model.audio_projection(model.audio_branch(wav.to(device))), dim=-1)
        for style, text_emb in text_embs.items():
            predictions[style].append(candidates[int((audio_emb @ text_emb.t()).argmax())])
        targets.append(label[0])
        if i % 500 == 0:
            print(f"sample {i}/{len(loader)}", flush=True)

    results = {}
    for style in queries:
        results[style] = {
            "ACC": audmetric.accuracy(targets, predictions[style]),
            "UAR": audmetric.unweighted_average_recall(targets, predictions[style]),
            "F1": audmetric.unweighted_average_fscore(targets, predictions[style]),
        }
        print(f"--- {name} | {style} ---")
        for label in candidates:
            total = sum(t == label for t in targets)
            correct = sum(t == label and p == label
                          for t, p in zip(targets, predictions[style]))
            print(f"{label:>20}: {correct:5d}/{total:5d}  recall {correct / max(total, 1):.3f}")
        print(yaml.dump(results[style]), flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"results.{name}.zsl.yaml"), "w") as fp:
        yaml.dump(results, fp)
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = load_model(os.path.join(CKPT_DIR, "best.pth.tar"), device)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    for name, cfg in DATASETS.items():
        if not (os.path.exists(cfg["wav_root"]) and os.path.exists(cfg["manifest"])):
            print(f"{name}: skipped (set wav_root and manifest in DATASETS)", flush=True)
            continue
        evaluate_zero_shot(name, cfg["wav_root"], cfg["manifest"], model, tokenizer,
                           device, cfg.get("template", SENTENCE_TEMPLATE))
