import os

import opensmile

from adapter import load_database


def extract_features(dataset_root, db=None, num_workers=8, limit=None):
    if db is None:
        db = load_database(dataset_root)

    files = db.files
    if limit is not None:
        files = files[:limit]

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_workers=num_workers,
        verbose=True,
    )
    features = smile.process_index(files, root=os.path.join(dataset_root, "original"))

    features = features.reset_index()
    features["Duration"] = (features["end"] - features["start"]).dt.total_seconds()
    features = features.drop(["start", "end"], axis=1)
    return features


if __name__ == "__main__":
    dataset_root = "/nas/student/DavidFrank/MSP-Podcast"
    out_csv = "features.csv"
    feats = extract_features(dataset_root, num_workers=8, limit=None)
    feats.to_csv(out_csv, index=False)
    print(f"Wrote {len(feats)} feature rows to {out_csv}")
