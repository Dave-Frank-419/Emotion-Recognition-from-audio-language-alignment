import os

import pandas as pd

RAVDESS_EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
TESS_RENAME = {"ps": "pleasant surprise", "sad": "sadness"}


def write_manifest(rows, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df = pd.DataFrame(rows, columns=["file", "label"])
    df.to_csv(dest, index=False)
    print(f"{dest}: {len(df)} files | {df['label'].value_counts().to_dict()}")


def ravdess(wav_root, dest="manifests/ravdess.csv"):
    rows = []
    for dirpath, _, files in os.walk(wav_root):
        for f in sorted(files):
            if f.endswith(".wav") and f.startswith("03-01"):
                rows.append((os.path.relpath(os.path.join(dirpath, f), wav_root),
                             RAVDESS_EMOTIONS[f.split("-")[2]]))
    write_manifest(rows, dest)


def tess(wav_root, dest="manifests/tess.csv"):
    rows = []
    for dirpath, _, files in os.walk(wav_root):
        for f in sorted(files):
            if f.endswith(".wav"):
                label = f[:-4].split("_")[-1].lower()
                rows.append((os.path.relpath(os.path.join(dirpath, f), wav_root),
                             TESS_RENAME.get(label, label)))
    write_manifest(rows, dest)


if __name__ == "__main__":
    ravdess("/data/chi-gpu1/ge96xah/data/RAVDESS")
    tess("/data/chi-gpu1/ge96xah/data/TESS")
