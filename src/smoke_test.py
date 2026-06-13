import json
import os

import opensmile
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from adapter import DEFAULT_EMOTIONS, audio_root, build_label_dataframe, load_database
from template_creation import create_templates_fix
from llm_caption_generator import LLMCaptioner
from ParaCLAP import (AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM,
                      CLAP, CaptionDataset, build_optimizer, clip_loss)

DATASET_ROOT = "/nas/student/DavidFrank/MSP-Podcast"
CAPTION_DIR = "captions_smoke"
N = 16

QUANTILE_COLS = [
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "loudness_sma3_amean",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "equivalentSoundLevel_dBp",
    "Duration",
]
TEMPLATE_INPUTS = ["emotion", "gender", "pitch", "intensity", "duration", "jitter", "shimmer"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    db = load_database(DATASET_ROOT)
    labels = db["categories.consensus.train"].df
    labels = labels[labels["emotion"].isin(DEFAULT_EMOTIONS)].head(N)
    labels = pd.concat([labels, db["speaker"].get(index=labels.index)], axis=1)
    speaker_labels = db.schemes["speaker"].labels
    labels["gender"] = labels["speaker"].map(
        lambda s: speaker_labels.get(s, {}).get("gender"), na_action="ignore")
    print(f"[1] labels ok: {len(labels)} rows, columns {labels.columns.tolist()}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    feats = smile.process_index(labels.index, root=os.path.join(DATASET_ROOT, "original"))
    feats = feats.reset_index()
    feats["Duration"] = (feats["end"] - feats["start"]).dt.total_seconds()
    feats = feats.drop(["start", "end"], axis=1).set_index("file")
    for c in QUANTILE_COLS:
        qt = QuantileTransformer(n_quantiles=min(len(feats), 100))
        feats[f"{c}.quantile"] = qt.fit_transform(feats[[c]])
    df = pd.concat([labels, feats.reindex(labels.index)], axis=1)
    print(f"[2] features ok: {feats.shape[1]} columns")

    df["templates"] = df.apply(lambda r: create_templates_fix(r, TEMPLATE_INPUTS), axis=1)
    print(f"[3] templates ok, e.g. {df['templates'].iloc[0]}")

    captioner = LLMCaptioner()
    items = [(os.path.basename(i), r["emotion"], r["gender"], r["templates"])
             for i, r in df.iterrows()]
    captions = captioner.caption_batch(items)
    print(f"[4] captions ok, e.g. {captions[0]!r}")

    os.makedirs(CAPTION_DIR, exist_ok=True)
    for (i, _), cap in zip(df.iterrows(), captions):
        with open(os.path.join(CAPTION_DIR, f"{os.path.basename(i)}.json"), "w",
                  encoding="utf-8") as fp:
            json.dump({"caption": cap}, fp, ensure_ascii=False)

    del captioner
    if device == "cuda":
        torch.cuda.empty_cache()

    df_tr = build_label_dataframe(DATASET_ROOT, "train", emotions=DEFAULT_EMOTIONS, db=db)
    keep = {os.path.basename(i) for i in df.index}
    df_tr = df_tr[df_tr.index.isin(keep)]
    loader = DataLoader(CaptionDataset(df_tr, audio_root(DATASET_ROOT), CAPTION_DIR),
                        batch_size=8, shuffle=True)
    wav, caps = next(iter(loader))

    model = CLAP(AUDIO_ENCODER, TEXT_ENCODER, EMBEDDING_DIM).to(device)
    model.train()
    optimizer = build_optimizer(model)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
    tokens = tokenizer(list(caps), padding=True, truncation=True, return_tensors="pt").to(device)

    audio, text, scale = model(wav.to(device), tokens)
    loss = clip_loss(audio, text, scale)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    audio_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model.audio_branch.parameters())
    text_grads = any(p.grad is not None for p in model.text_branch.parameters())
    print(f"[5] train step ok: loss={loss.item():.4f}, "
          f"audio_grads={audio_grads}, text_grads={text_grads}")
    print("smoke test passed")


if __name__ == "__main__":
    main()
