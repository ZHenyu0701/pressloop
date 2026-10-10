import argparse
import os

from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        raise ValueError(f"Unsupported feature file: {path}")

    feat = to_numpy(feat)
    feat = np.array(feat, dtype=np.float32)
    if feat.ndim == 1:
        feat = feat.reshape(-1, 1)
    elif feat.ndim > 2:
        feat = feat.reshape(feat.shape[0], -1)
    return feat


def append_optional_features(X, feature_path=None):
    if feature_path is None:
        return X

    feat = load_feature_vector(feature_path)
    if len(feat) != len(X):
        raise ValueError(
            f"Feature length mismatch: len(features)={len(feat)}, len(X)={len(X)}"
        )

    X = np.concatenate([X, feat], axis=1)
    print("optional feature shape:", feat.shape)
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
        "--output_dir",
        type=str,
        default="/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1/sklearn_output",
    )
    parser.add_argument(
        "--heart_rate_path",
        type=str,
        default=None,
        help="Optional .pt or .npy feature file aligned 1:1 with embeddings.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint directory containing config.json. Required for auto-generating WESAD HR features.",
    )
    parser.add_argument(
        "--wesad_raw_dir",
        type=str,
        default=None,
        help="Root WESAD raw directory containing Sx/Sx.pkl.",
    )
    parser.add_argument(
        "--signal_source",
        type=str,
        default="bvp",
        choices=["bvp", "ecg"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def maybe_prepare_hr_features(args):
    if args.heart_rate_path is not None:
        return args.heart_rate_path

    if args.wesad_raw_dir is None:
        return None

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
    print("auto-generated heart-rate features:", heart_rate_path)
    return heart_rate_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    heart_rate_path = maybe_prepare_hr_features(args)

    X = torch.load(args.embeddings_path, map_location="cpu")
    y = torch.load(args.labels_path, map_location="cpu")

    X = to_numpy(X)
    y = squeeze_labels(y)
    X = append_optional_features(X, heart_rate_path)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("labels unique:", np.unique(y))

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print("\nTrain/Test split:")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(args.output_dir, "scaler.joblib")
    dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=args.random_state,
            class_weight="balanced",
        ),
        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=args.random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=args.random_state,
            class_weight="balanced",
        ),
    }

    results = []
    confusion_matrices = {}
    roc_data = {}

    for name, model in models.items():
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(cm)

        model_path = os.path.join(args.output_dir, f"{name}.joblib")
        dump(model, model_path)
        print(f"Model saved to: {model_path}")

        report_path = os.path.join(args.output_dir, f"{name}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred, zero_division=0))
        print(f"Classification report saved to: {report_path}")

        result_row = {
            "Model": name,
            "Accuracy": acc,
            "F1": f1,
            "Precision": precision,
            "Recall": recall,
        }

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_data[name] = (fpr, tpr, roc_auc)
            result_row["AUC"] = roc_auc
            print(f"AUC      : {roc_auc:.4f}")
        else:
            result_row["AUC"] = np.nan

        results.append(result_row)
        confusion_matrices[name] = cm

    results_df = pd.DataFrame(results)
    summary_path = os.path.join(args.output_dir, "model_comparison.csv")
    results_df.to_csv(summary_path, index=False)

    print("\n===== Summary =====")
    print(results_df)
    print(f"\nSummary saved to: {summary_path}")

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.ylabel("Accuracy")
    plt.title("Model Comparison - Accuracy")
    plt.ylim(0.0, 1.0)
    for i, v in enumerate(results_df["Accuracy"]):
        plt.text(i, min(v + 0.01, 0.98), f"{v:.3f}", ha="center")
    plt.tight_layout()
    acc_fig_path = os.path.join(args.output_dir, "accuracy_comparison.png")
    plt.savefig(acc_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Accuracy comparison figure saved to: {acc_fig_path}")

    x = np.arange(len(results_df))
    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, results_df["F1"], width=width, label="F1")
    plt.bar(x, results_df["Precision"], width=width, label="Precision")
    plt.bar(x + width, results_df["Recall"], width=width, label="Recall")
    plt.xticks(x, results_df["Model"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Comparison - F1 / Precision / Recall")
    plt.legend()
    plt.tight_layout()
    f1_fig_path = os.path.join(args.output_dir, "f1_precision_recall_comparison.png")
    plt.savefig(f1_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"F1/Precision/Recall figure saved to: {f1_fig_path}")

    for name, cm in confusion_matrices.items():
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {name}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No Stress", "Stress"])
        plt.yticks(tick_marks, ["No Stress", "Stress"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_fig_path = os.path.join(args.output_dir, f"confusion_matrix_{name}.png")
        plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to: {cm_fig_path}")

    if len(roc_data) > 0:
        plt.figure(figsize=(6, 5))
        for name, (fpr, tpr, roc_auc) in roc_data.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        roc_fig_path = os.path.join(args.output_dir, "roc_curve.png")
        plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"ROC curve saved to: {roc_fig_path}")
    else:
        print("No ROC data available.")

    print("\nDone.")


if __name__ == "__main__":
    main()
