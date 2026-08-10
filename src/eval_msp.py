import os

import audformat
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


class FullAudioDataset(Dataset):

    def __init__(self, df, wav_root):
        self.files = list(df.index)
        self.emotions = list(df["emotion"])
        self.wav_root = wav_root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        wav, sr = torchaudio.load(os.path.join(self.wav_root, self.files[index]))
        if sr != SAMPLE_RATE:
            wav = F_audio.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
        return wav.mean(0), self.emotions[index]


def evaluate_msp(dataset_root, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    db = audformat.Database.load(os.path.join(dataset_root, "converted"))
    df_test = db["categories.consensus.test1"].df
    df_test = df_test.reset_index()
    df_test["file"] = df_test["file"].apply(os.path.basename)
    df_test = df_test.set_index("file")

    candidates = list(df_test["emotion"].unique())
    print(f"test1: {len(df_test)} utterances | candidates: {candidates}")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    candidate_tokens = tokenizer(candidates, padding=True, truncation=True,
                                 return_tensors="pt").to(device)

    model = CLAP(AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']})")
    model.to(device).eval()

    loader = DataLoader(FullAudioDataset(df_test, os.path.join(dataset_root, "original", "Audios")),
                        batch_size=1, num_workers=4, pin_memory=True)

    targets, predictions = [], []
    for i, (wav, emotion) in enumerate(loader):
        with torch.no_grad():
            audio_emb, text_emb, scale = model(wav.to(device), candidate_tokens)
            similarity = scale * F.normalize(audio_emb, dim=-1) @ F.normalize(text_emb, dim=-1).t()
        predictions.append(candidates[int(similarity.argmax())])
        targets.append(emotion[0])
        if i % 500 == 0:
            print(f"sample {i}/{len(loader)}", flush=True)

    results = {
        "ACC": audmetric.accuracy(targets, predictions),
        "UAR": audmetric.unweighted_average_recall(targets, predictions),
        "F1": audmetric.unweighted_average_fscore(targets, predictions),
    }
    print(yaml.dump(results))
    for emotion in candidates:
        total = sum(t == emotion for t in targets)
        correct = sum(t == emotion and p == emotion for t, p in zip(targets, predictions))
        print(f"{emotion:>12}: {correct:5d}/{total:5d}  recall {correct / max(total, 1):.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.msp.zsl.yaml"), "w") as fp:
        yaml.dump(results, fp)
    pd.DataFrame({"prediction": predictions, "emotion": targets}, index=df_test.index) \
        .reset_index().to_csv(os.path.join(RESULTS_DIR, "results.msp.zsl.csv"), index=False)
    return results


if __name__ == "__main__":
    evaluate_msp(
        dataset_root="/nas/student/DavidFrank/MSP-Podcast",
        ckpt_path=os.path.join(CKPT_DIR, "best.pth.tar"),
    )
