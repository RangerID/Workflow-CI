import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Default konstanta ─────────────────────────────────────────────────────────
DATA_DIR        = "plants_preprocessing"
EXPERIMENT_NAME = "plants_classification_ci"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple:
    """Memuat array numpy hasil preprocessing."""
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    X_valid = np.load(os.path.join(data_dir, "X_valid.npy"))
    y_valid = np.load(os.path.join(data_dir, "y_valid.npy"))
    label_df    = pd.read_csv(os.path.join(data_dir, "label_encoder.csv"))
    class_names = label_df.sort_values("label_index")["class"].tolist()
    logger.info("Data loaded — Train:%s | Test:%s | Valid:%s",
                X_train.shape, X_test.shape, X_valid.shape)
    return X_train, y_train, X_test, y_test, X_valid, y_valid, class_names


def save_confusion_matrix(y_true, y_pred, class_names, path="confusion_matrix.png"):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def build_pipeline(model_name: str, n_estimators: int,
                   max_depth, C: float) -> Pipeline:
    md = None if max_depth == 0 else int(max_depth)

    if model_name == "RandomForest":
        return Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=md,
                random_state=42,
                n_jobs=-1,
            ))
        ])
    elif model_name == "LogisticRegression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=C, max_iter=500, random_state=42,
            ))
        ])
    elif model_name == "SVM":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=C, random_state=42, probability=True))
        ])
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}. "
                         "Pilih: RandomForest | LogisticRegression | SVM")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args) -> None:

    # ── Setup experiment ────────────────────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{args.model}_CI"):

        # ── Autolog ─────────────────────────────────────────────────────────
        mlflow.sklearn.autolog(
            log_input_examples=False,
            log_model_signatures=True,
            log_models=True,
        )

        # ── Load data ────────────────────────────────────────────────────────
        X_train, y_train, X_test, y_test, X_valid, y_valid, class_names = \
            load_data(args.data_dir)

        # ── Build & train ────────────────────────────────────────────────────
        pipeline = build_pipeline(
            args.model, args.n_estimators, args.max_depth, args.C
        )
        logger.info("Training model: %s", args.model)
        pipeline.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────────────────────────────
        y_pred_test  = pipeline.predict(X_test)
        y_pred_valid = pipeline.predict(X_valid)

        test_acc  = accuracy_score(y_test,  y_pred_test)
        valid_acc = accuracy_score(y_valid, y_pred_valid)
        test_f1   = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
        test_prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        test_rec  = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)

        # ── Manual log (test/valid tidak ter-cover autolog) ──────────────────
        mlflow.log_metric("test_accuracy",          test_acc)
        mlflow.log_metric("valid_accuracy",         valid_acc)
        mlflow.log_metric("test_f1_weighted",       test_f1)
        mlflow.log_metric("test_precision_weighted",test_prec)
        mlflow.log_metric("test_recall_weighted",   test_rec)

        # ── Tags ─────────────────────────────────────────────────────────────
        mlflow.set_tag("model_type",       args.model)
        mlflow.set_tag("dataset",          "plants_type")
        mlflow.set_tag("triggered_by",     os.getenv("GITHUB_EVENT_NAME", "manual"))
        mlflow.set_tag("github_sha",       os.getenv("GITHUB_SHA", "local")[:8])
        mlflow.set_tag("github_actor",     os.getenv("GITHUB_ACTOR", "local"))

        # ── Artifacts ────────────────────────────────────────────────────────
        # 1. Confusion matrix
        cm_path = save_confusion_matrix(y_test, y_pred_test, class_names)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        os.remove(cm_path)

        # 2. Classification report
        rpt = classification_report(
            y_test, y_pred_test, target_names=class_names, output_dict=True
        )
        rpt_df = pd.DataFrame(rpt).transpose()
        rpt_df.to_csv("classification_report.csv")
        mlflow.log_artifact("classification_report.csv", artifact_path="reports")
        os.remove("classification_report.csv")

        # ── Print summary ─────────────────────────────────────────────────────
        logger.info("=" * 55)
        logger.info("Model          : %s", args.model)
        logger.info("test_accuracy  : %.4f", test_acc)
        logger.info("valid_accuracy : %.4f", valid_acc)
        logger.info("test_f1        : %.4f", test_f1)
        logger.info("=" * 55)
        print(classification_report(y_test, y_pred_test, target_names=class_names))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MLflow Project – Plants Type Training")
    p.add_argument("--model",        type=str,   default="RandomForest",
                   choices=["RandomForest", "LogisticRegression", "SVM"],
                   help="Algoritma yang digunakan")
    p.add_argument("--n_estimators", type=int,   default=100,
                   help="Jumlah pohon (RandomForest)")
    p.add_argument("--max_depth",    type=int,   default=0,
                   help="Kedalaman maksimum pohon; 0 = None (RandomForest)")
    p.add_argument("--C",            type=float, default=1.0,
                   help="Regularisasi C (LogisticRegression / SVM)")
    p.add_argument("--data_dir",     type=str,   default=DATA_DIR,
                   help="Direktori data preprocessing")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
