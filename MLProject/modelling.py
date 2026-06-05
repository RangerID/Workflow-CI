import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

# ── Konfigurasi path data preprocessed ───────────────────────────────────────
PROCESSED_DIR = os.path.join(
    os.path.dirname(__file__), "Eksperimen_SML"
)

# Split sesuai nama folder dataset
TRAIN_SPLIT = "Train_Set_Folder"
TEST_SPLIT  = "Test_Set_Folder"
VALID_SPLIT = "Validation_Set_Folder"


# ── 1. Load data preprocessed ────────────────────────────────────────────────

def load_split(processed_dir: str, split_name: str):
    x_path = os.path.join(processed_dir, f"X_{split_name}.npy")
    y_path = os.path.join(processed_dir, f"y_{split_name}.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {x_path} atau {y_path}\n"
            "Pastikan sudah menjalankan automate_Nama-siswa.py terlebih dahulu."
        )

    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def flatten_images(X: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


# ── 2. Main pipeline ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Plants Type Classification – MLflow Tracking")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\n[1/4] Memuat data dari: {PROCESSED_DIR}")
    X_train, y_train = load_split(PROCESSED_DIR, TRAIN_SPLIT)
    X_test,  y_test  = load_split(PROCESSED_DIR, TEST_SPLIT)
    X_val,   y_val   = load_split(PROCESSED_DIR, VALID_SPLIT)

    print(f"  Train  : X={X_train.shape}, y={y_train.shape}")
    print(f"  Test   : X={X_test.shape},  y={y_test.shape}")
    print(f"  Valid  : X={X_val.shape},   y={y_val.shape}")

    # Flatten (N, 128, 128, 3) → (N, 49152)
    X_train_flat = flatten_images(X_train)
    X_test_flat  = flatten_images(X_test)
    X_val_flat   = flatten_images(X_val)

    # ── Load label encoder ─────────────────────────────────────────────────────
    enc_path = os.path.join(PROCESSED_DIR, "label_encoder.csv")
    if os.path.exists(enc_path):
        df_enc     = pd.read_csv(enc_path)
        class_names = df_enc.sort_values("label_index")["class"].tolist()
    else:
        class_names = [str(i) for i in sorted(np.unique(y_train))]

    n_classes = len(class_names)
    print(f"\n  Jumlah kelas : {n_classes}")
    print(f"  Nama kelas   : {class_names}")

    # ── Konfigurasi MLflow ─────────────────────────────────────────────────────
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("plants_type_classification")

    # ── Aktifkan autolog ───────────────────────────────────────────────────────
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_models=True,
        silent=False,
    )

    # ── Training (tanpa hyperparameter tuning) ─────────────────────────────────
    print("\n[2/4] Melatih RandomForestClassifier ...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    # MLflow autolog menangkap fit() secara otomatis
    with mlflow.start_run(run_name="random_forest_baseline"):
        model.fit(X_train_flat, y_train)

        # ── Evaluasi manual (tambahan) ─────────────────────────────────────────
        print("\n[3/4] Evaluasi model ...")

        # Test set
        y_pred_test = model.predict(X_test_flat)
        acc_test    = accuracy_score(y_test, y_pred_test)

        # Validation set
        y_pred_val = model.predict(X_val_flat)
        acc_val    = accuracy_score(y_val, y_pred_val)

        # Log metrik tambahan secara manual
        mlflow.log_metric("test_accuracy",  acc_test)
        mlflow.log_metric("valid_accuracy", acc_val)

        print(f"\n  Accuracy (Test)       : {acc_test:.4f}")
        print(f"  Accuracy (Validation) : {acc_val:.4f}")

        # Classification report → simpan sebagai artefak teks
        report_test = classification_report(
            y_test, y_pred_test, target_names=class_names
        )
        report_path = os.path.join(os.path.dirname(__file__), "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("=== Test Set ===\n")
            f.write(report_test)
            f.write("\n\n=== Validation Set ===\n")
            f.write(
                classification_report(y_val, y_pred_val, target_names=class_names)
            )
        mlflow.log_artifact(report_path)

        print("\n[4/4] Run MLflow selesai.")
        run_id = mlflow.active_run().info.run_id
        print(f"  Run ID   : {run_id}")
        print(f"  Database : {db_path}")

    print("\n" + "=" * 60)
    print("✅  Training selesai.")
    print("=" * 60)


if __name__ == "__main__":
    main()