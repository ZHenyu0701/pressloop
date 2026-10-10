import argparse
import os
from joblib import dump

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from wesad_hr_features import export_wesad_hr_features


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)


def squeeze_labels(y):
    y = to_numpy(y)
    y = np.array(y)

    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    if y.ndim > 1:
        y = y.reshape(-1)

    return y


def load_feature_vector(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        feat = torch.load(path, map_location="cpu")
    elif ext == ".npy":
        feat = np.load(path)
    else:
        raise ValueError(f"Unsupported heart-rate feature file: {path}")

    feat = to_numpy(feat)
    feat = np.array(feat, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat.reshape(-1, 1)
    elif feat.ndim > 2:
        feat = feat.reshape(feat.shape[0], -1)
    return feat


def append_optional_features(X, heart_rate_path=None):
    if heart_rate_path is None:
        return X

    heart_rate = load_feature_vector(heart_rate_path)
    if len(heart_rate) != len(X):
        raise ValueError(
            f"Heart-rate feature length mismatch: len(hr)={len(heart_rate)}, len(X)={len(X)}"
        )

    X = np.concatenate([X, heart_rate], axis=1)
    print("heart-rate feature shape:", heart_rate.shape)
    return X


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1/embeddings.pt",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1/labels.pt",
    )
    parser.add_argument(
        "--heart_rate_path",
        type=str,
        default=None,
        help="Optional .pt or .npy file aligned 1:1 with embeddings. Shape can be [N] or [N, 1].",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint directory containing config.json. Used to auto-export aligned WESAD heart-rate features.",
    )
    parser.add_argument(
        "--wesad_raw_dir",
        type=str,
        default=None,
        help="Root WESAD raw directory containing Sx/Sx.pkl. If provided without --heart_rate_path, HR features are generated automatically.",
    )
    parser.add_argument(
        "--signal_source",
        type=str,
        default="bvp",
        choices=["bvp", "ecg"],
        help="Signal used to derive heart rate from raw WESAD.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split used when auto-generating WESAD heart-rate features.",
    )
    parser.add_argument("--output_dir", type=str, default="sklearn_output")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.heart_rate_path is None and args.wesad_raw_dir is not None:
        if args.ckpt_dir is None:
            raise ValueError("--ckpt_dir is required when auto-generating heart-rate features")
        config_path = os.path.join(args.ckpt_dir, "config.json")
        heart_rate_path = os.path.join(args.ckpt_dir, f"heart_rate_{args.split}_{args.signal_source}.pt")
        export_wesad_hr_features(
            config_path=config_path,
            raw_data_dir=args.wesad_raw_dir,
            split=args.split,
            out_path=heart_rate_path,
            signal_source=args.signal_source,
        )
        args.heart_rate_path = heart_rate_path
        print("auto-generated heart-rate features:", args.heart_rate_path)

    X = torch.load(args.embeddings_path, map_location="cpu")
    y = torch.load(args.labels_path, map_location="cpu")

    X = to_numpy(X)
    y = squeeze_labels(y)
    X = append_optional_features(X, args.heart_rate_path)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("labels unique:", np.unique(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    y_pred = clf.predict(X_scaled)
    acc = accuracy_score(y, y_pred)

    print(f"\nTrain Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    model_path = os.path.join(args.output_dir, "final_model.joblib")
    scaler_path = os.path.join(args.output_dir, "scaler.joblib")

    dump(clf, model_path)
    dump(scaler, scaler_path)

    print("\n保存完成：")
    print(model_path)
    print(scaler_path)


if __name__ == "__main__":
    main()
