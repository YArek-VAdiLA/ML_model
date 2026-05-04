from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_CSV = PROJECT_ROOT / "dataset_train.csv"
TEST_CSV = PROJECT_ROOT / "dataset_test.csv"

SVM_MODEL_PATH = PROJECT_ROOT / "gesture_svm.joblib"
RF_MODEL_PATH = PROJECT_ROOT / "gesture_rf.joblib"
LABEL_ENCODER_PATH = PROJECT_ROOT / "label_encoder.joblib"

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=["label"])
    y = df["label"]

    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n" + "=" * 70)
    print(f"{name}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 70 + "\n")

    return acc


def main():
    print("[INFO] Загрузка train/test...")
    X_train, y_train = load_data(TRAIN_CSV)
    X_test, y_test = load_data(TEST_CSV)

    X_train = X_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)

    print(f"[INFO] Train shape: {X_train.shape}")
    print(f"[INFO] Test shape:  {X_test.shape}")

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    print("[INFO] Классы:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i}: {cls}")

    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=42
        ))
    ])

    print("\n[INFO] Обучение SVM...")
    svm_model.fit(X_train, y_train_enc)

    svm_acc = evaluate_model(
        "SVM",
        svm_model,
        X_test,
        y_test_enc
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Обучение Random Forest...")
    rf_model.fit(X_train, y_train_enc)

    rf_acc = evaluate_model(
        "Random Forest",
        rf_model,
        X_test,
        y_test_enc
    )

    joblib.dump(svm_model, SVM_MODEL_PATH)
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print("[INFO] Модели сохранены:")
    print(f"  {SVM_MODEL_PATH.name}")
    print(f"  {RF_MODEL_PATH.name}")
    print(f"  {LABEL_ENCODER_PATH.name}")

    if svm_acc >= rf_acc:
        print(f"\n[RESULT] Лучшая модель сейчас: SVM ({svm_acc:.4f})")
    else:
        print(f"\n[RESULT] Лучшая модель сейчас: Random Forest ({rf_acc:.4f})")


if __name__ == "__main__":
    main()