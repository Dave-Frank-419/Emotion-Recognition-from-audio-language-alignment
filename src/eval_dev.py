import os

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as F_audio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from adapter import DEFAULT_EMOTIONS, audio_root, build_label_dataframe
from ParaCLAP import AUDIO_ENCODER, CKPT_DIR, CLAP, CLIP_SECONDS, EMBEDDING_DIM, SAMPLE_RATE, TEXT_ENCODER

SPLIT = "validation"
TEXT_QUERY = "speaker is {}"
BATCH_SIZE = 32


class EvalDataset(Dataset):

    def __init__(self, df, wav_root):
        self.files = list(df.index)
        self.labels = [DEFAULT_EMOTIONS.index(e) for e in df["emotion"]]
        self.wav_root = wav_root
        self.target_length = CLIP_SECONDS * SAMPLE_RATE

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            wav, sr = torchaudio.load(os.path.join(self.wav_root, self.files[index]))
        except Exception:
            print(f"audio load failed, using silence: {self.files[index]}", flush=True)
            return torch.zeros(self.target_length), self.labels[index]
        if sr != SAMPLE_RATE:
            wav = F_audio.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
        wav = wav.mean(0, keepdim=True)
        n = wav.shape[1]
        if n > self.target_length:
            start = (n - self.target_length) // 2
            wav = wav[:, start:start + self.target_length]
        else:
            wav = F.pad(wav, (0, self.target_length - n))
        return wav.squeeze(0), self.labels[index]


def score(model, pooled, text_cls, labels, device, tag):
    with torch.inference_mode():
        audio_emb = model.audio_projection(pooled.to(device))
        text_emb = model.text_projection(text_cls)
        pred = (audio_emb @ text_emb.t()).argmax(dim=1).cpu()

    print(f"--- {tag} ---")
    recalls = []
    for k, emotion in enumerate(DEFAULT_EMOTIONS):
        mask = labels == k
        if mask.any():
            correct = int((pred[mask] == k).sum())
            recalls.append(correct / int(mask.sum()))
            print(f"{emotion:>10}: {correct:5d}/{int(mask.sum()):5d}  recall {recalls[-1]:.3f}")
    uar = sum(recalls) / len(recalls)
    acc = (pred == labels).float().mean().item()
    print(f"{tag}: UAR {uar:.4f} | acc {acc:.4f} | chance {1 / len(recalls):.4f}", flush=True)


def evaluate(dataset_root, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df = build_label_dataframe(dataset_root, split=SPLIT, emotions=DEFAULT_EMOTIONS)
    print(f"{SPLIT}: {len(df)} utterances | query '{TEXT_QUERY}'")

    model = CLAP(AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    tokens = tokenizer([TEXT_QUERY.format(e) for e in DEFAULT_EMOTIONS],
                       padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        text_cls = model.text_branch(tokens)

    loader = DataLoader(EvalDataset(df, audio_root(dataset_root)),
                        batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    pooled, labels = [], []
    with torch.inference_mode():
        for i, (wav, y) in enumerate(loader):
            pooled.append(model.audio_branch(wav.to(device)).cpu())
            labels.append(y)
            if i % 50 == 0:
                print(f"encoding batch {i}/{len(loader)}", flush=True)
    pooled, labels = torch.cat(pooled), torch.cat(labels)

    score(model, pooled, text_cls, labels, device, "untrained")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']}, train loss {ckpt['best_loss']:.4f})")
    score(model, pooled, text_cls, labels, device, "trained")


if __name__ == "__main__":
    evaluate(
        dataset_root="/nas/student/DavidFrank/MSP-Podcast",
        ckpt_path=os.path.join(CKPT_DIR, "best.pth.tar"),
    )
