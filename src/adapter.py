import os

import audformat
import pandas as pd

DEFAULT_EMOTIONS = [
    "neutral", "fear", "sadness", "disgust", "happiness",
    "other", "anger", "contempt", "surprise",
]


def audio_root(dataset_root):
    return os.path.join(dataset_root, "original", "Audios")


def load_database(dataset_root):
    return audformat.Database.load(os.path.join(dataset_root, "converted"))


def build_label_dataframe(dataset_root, split="train", emotions=None, db=None):
    if db is None:
        db = load_database(dataset_root)

    df = db[f"categories.consensus.{split}"].df
    df = pd.concat((df, db[f"dimensions.consensus.{split}"].get(index=df.index)), axis=1)
    df = pd.concat((df, db["speaker"].get(index=df.index)), axis=1)
    speaker_labels = db.schemes["speaker"].labels
    df["gender"] = df["speaker"].map(
        lambda spk: speaker_labels.get(spk, {}).get("gender"), na_action="ignore")

    if emotions is not None:
        df = df.loc[df["emotion"].isin(list(emotions))]

    df = df.reset_index()
    df["file"] = df["file"].apply(os.path.basename)
    return df.set_index("file")
