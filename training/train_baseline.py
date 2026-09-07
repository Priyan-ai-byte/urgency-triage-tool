"""
train_baseline.py

Stage 2 deliverable: trains and evaluates the first baseline ML model
(TF-IDF + Logistic Regression) for the Human-Reviewed Urgency Triage
Tool.

This script:
  1. Loads the synthetic dataset
  2. Validates required columns are present
  3. Splits into a stratified train/test split
  4. Fits the text + structured feature pipeline (fit ONLY on the
     training split, to avoid leakage)
  5. Trains a Logistic Regression classifier
  6. Evaluates on the held-out test split
  7. Prints a full metrics report
  8. Saves the fitted pipeline to models/baseline_model.joblib
  9. Writes the measured results to evaluation/baseline_results.md

IMPORTANT SAFETY NOTE: this script trains a baseline classifier only.
It does not implement, and must never be treated as, an autonomous
triage decision system. See README.md for the full safety statement.
NEEDS_CLARIFICATION is intentionally NOT one of the training classes -
that is a separate uncertainty/safety layer to be added in a later
stage, sitting on top of this model's output.

Run from the project root:
    python training/train_baseline.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow "from src.features import ..." to work when this script is run
# directly (python training/train_baseline.py) from the project root.
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    CLASS_LABELS,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    URGENT_LABELS,
    build_pipeline,
)

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_intake_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "baseline_results.md"

TEST_SIZE = 0.20

REQUIRED_COLUMNS = [
    "request_id",
    "intake_message",
    "risk_indicators",
    "waiting_time_minutes",
    "channel",
    "professional_triage_outcome",
    "urgency_label",
]


def load_and_validate_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run data/generate_synthetic_data.py first."
        )
    df = pd.read_csv(DATA_PATH)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    print(f"Loaded dataset: {len(df)} rows from {DATA_PATH}")
    return df


def urgent_vs_not(labels) -> np.ndarray:
    """
    Map 4-class labels to a binary URGENT / NOT_URGENT view, used ONLY
    for the analysis-only urgent-recall/precision/false-positive-rate
    metrics below. The classifier itself is never trained on this binary
    target - it is purely a post-hoc lens on the 4-class predictions.
    """
    return np.array(["URGENT" if lab in URGENT_LABELS else "NOT_URGENT" for lab in labels])


def compute_urgent_metrics(y_true, y_pred) -> dict:
    true_bin = urgent_vs_not(y_true)
    pred_bin = urgent_vs_not(y_pred)

    tp = int(np.sum((true_bin == "URGENT") & (pred_bin == "URGENT")))
    fn = int(np.sum((true_bin == "URGENT") & (pred_bin == "NOT_URGENT")))
    fp = int(np.sum((true_bin == "NOT_URGENT") & (pred_bin == "URGENT")))
    tn = int(np.sum((true_bin == "NOT_URGENT") & (pred_bin == "NOT_URGENT")))

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "urgent_recall": recall,
        "urgent_precision": precision,
        "urgent_false_positive_rate": fpr,
    }


def format_confusion_matrix(cm: np.ndarray, labels) -> str:
    header = "            " + "".join(f"{l:>12s}" for l in labels)
    lines = [header]
    for i, row_label in enumerate(labels):
        row = "".join(f"{cm[i, j]:12d}" for j in range(len(labels)))
        lines.append(f"{row_label:12s}{row}")
    return "\n".join(lines)


def write_results_markdown(
    df,
    n_train,
    n_test,
    report_dict,
    report_text,
    cm,
    labels,
    accuracy,
    macro_f1,
    weighted_f1,
    urgent_metrics,
):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    class_counts = df["urgency_label"].value_counts().reindex(labels).to_dict()

    lines = []
    lines.append("# Baseline Model Results (TF-IDF + Logistic Regression)")
    lines.append("")
    lines.append(f"_Generated automatically by `training/train_baseline.py` on {timestamp}, "
                  "from actual measured output - no values below are invented._")
    lines.append("")
    lines.append("## Dataset Information")
    lines.append("")
    lines.append("- Source: `data/synthetic_intake_dataset.csv`")
    lines.append(f"- Total rows: {len(df)}")
    lines.append("- Class distribution (full dataset): "
                  + ", ".join(f"{k}={v}" for k, v in class_counts.items()))
    lines.append("")
    lines.append("## Train/Test Split")
    lines.append("")
    lines.append("- Split type: stratified")
    lines.append(f"- Test size: {TEST_SIZE:.0%}")
    lines.append(f"- Random state: {RANDOM_STATE}")
    lines.append(f"- Training rows: {n_train}")
    lines.append(f"- Test rows: {n_test}")
    lines.append("")
    lines.append("## Model Description")
    lines.append("")
    lines.append("- Algorithm: Logistic Regression (`class_weight='balanced'`, multinomial)")
    lines.append(f"- Random state: {RANDOM_STATE}")
    lines.append("- This is an intentionally simple, interpretable baseline. It has not been")
    lines.append("  tuned to hit any target metric - these are honest first-pass numbers, not")
    lines.append("  optimised results.")
    lines.append("")
    lines.append("## Feature Description")
    lines.append("")
    lines.append("- **Text**: TF-IDF over `intake_message` (unigrams + bigrams, English stopwords, min_df=2)")
    lines.append("- **Structured**: fixed binary encoding of `risk_indicators` (8 known categories + a")
    lines.append("  separate \"field was missing\" flag), plus `waiting_time_minutes` (median-imputed on")
    lines.append("  training data only, with its own missing-value flag, then scaled)")
    lines.append("- `professional_triage_outcome` and `urgency_label` are never used as inputs (target leakage check)")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"- **Accuracy**: {accuracy:.3f}")
    lines.append(f"- **Macro F1**: {macro_f1:.3f}")
    lines.append(f"- **Weighted F1**: {weighted_f1:.3f}")
    lines.append("")
    lines.append("## Per-Class Precision / Recall / F1")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1-score | Support |")
    lines.append("|---|---|---|---|---|")
    for label in labels:
        r = report_dict[label]
        lines.append(
            f"| {label} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1-score']:.3f} | {int(r['support'])} |"
        )
    lines.append("")
    lines.append("## Full Classification Report (scikit-learn text output)")
    lines.append("")
    lines.append("```")
    lines.append(report_text.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("Rows = true label, columns = predicted label.")
    lines.append("")
    lines.append("```")
    lines.append(format_confusion_matrix(cm, labels))
    lines.append("```")
    lines.append("")
    lines.append("## Urgent vs Not-Urgent Analysis (Analysis Only)")
    lines.append("")
    lines.append("`URGENT` = CRITICAL + HIGH, `NOT_URGENT` = MODERATE + ROUTINE. This collapses")
    lines.append("the model's normal 4-class predictions after the fact - no threshold tuning")
    lines.append("has been applied at this stage, and the model was never trained on this binary")
    lines.append("target.")
    lines.append("")
    lines.append(f"- True positives (urgent correctly flagged): {urgent_metrics['tp']}")
    lines.append(f"- False negatives (urgent missed): {urgent_metrics['fn']}")
    lines.append(f"- False positives (not-urgent flagged as urgent): {urgent_metrics['fp']}")
    lines.append(f"- True negatives (not-urgent correctly left alone): {urgent_metrics['tn']}")
    lines.append(f"- **Urgent recall**: {urgent_metrics['urgent_recall']:.3f}")
    lines.append(f"- **Urgent precision**: {urgent_metrics['urgent_precision']:.3f}")
    lines.append(f"- **Urgent false-positive rate**: {urgent_metrics['urgent_false_positive_rate']:.3f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This is an unoptimised first baseline, measured honestly before any tuning. It has "
        "not yet been compared against the target metrics from the project specification, and "
        "no confidence threshold or safety-override logic has been applied - both belong to a "
        "later evaluation stage. False positives and false negatives should be examined "
        "qualitatively (see `training/test_prediction.py` output and a future dedicated "
        "false-positive/false-negative analysis) before drawing any conclusions about "
        "real-world readiness."
    )
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote evaluation report to: {RESULTS_PATH}")


def main():
    df = load_and_validate_data()

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    labels = CLASS_LABELS  # fixed, documented order: CRITICAL, HIGH, MODERATE, ROUTINE

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")

    pipeline = build_pipeline()

    print("\nFitting pipeline on training data only (TF-IDF vocabulary and the waiting-time")
    print("median are both learned from X_train only, never from X_test)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report_dict = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    urgent_metrics = compute_urgent_metrics(y_test.to_numpy(), y_pred)

    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    print(f"\nAccuracy:     {accuracy:.3f}")
    print(f"Macro F1:     {macro_f1:.3f}")
    print(f"Weighted F1:  {weighted_f1:.3f}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix (rows=true, cols=pred):")
    print(format_confusion_matrix(cm, labels))
    print("\nUrgent (CRITICAL+HIGH) vs Not-Urgent analysis (analysis only, no tuning applied):")
    print(f"  Urgent recall:              {urgent_metrics['urgent_recall']:.3f}")
    print(f"  Urgent precision:           {urgent_metrics['urgent_precision']:.3f}")
    print(f"  Urgent false-positive rate: {urgent_metrics['urgent_false_positive_rate']:.3f}")
    print(
        f"  (TP={urgent_metrics['tp']}, FN={urgent_metrics['fn']}, "
        f"FP={urgent_metrics['fp']}, TN={urgent_metrics['tn']})"
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

    write_results_markdown(
        df,
        len(X_train),
        len(X_test),
        report_dict,
        report_text,
        cm,
        labels,
        accuracy,
        macro_f1,
        weighted_f1,
        urgent_metrics,
    )


if __name__ == "__main__":
    main()
