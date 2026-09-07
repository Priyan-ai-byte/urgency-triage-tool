"""
predict.py

Thin prediction interface around the saved baseline model pipeline.
Loads models/baseline_model.joblib and exposes predict_urgency(), which
takes a single raw intake record and returns the model's predicted
label and class probabilities.

IMPORTANT: this module deliberately does NOT implement any safety
override rules, uncertainty handling, evidence extraction, or human
review logic. It is a pure model-inference wrapper. Those safety-critical
layers must be added on top of this function in later project stages -
never folded into it - so this stays a clean, swappable component.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"

_model = None  # lazy-loaded, cached after first use


def load_model():
    """Load (and cache) the trained pipeline from disk."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run training/train_baseline.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_urgency(
    intake_message: str,
    risk_indicators: Optional[str] = None,
    waiting_time_minutes: Optional[float] = None,
) -> dict:
    """
    Run the baseline model on a single raw intake record.

    Parameters mirror the raw dataset columns. `risk_indicators` and
    `waiting_time_minutes` may be None - the underlying pipeline handles
    missing values safely (see src/features.py), so this does not need
    to guess or fill in default values itself.

    Returns a dictionary of the form:
        {
            "predicted_label": "HIGH",
            "class_probabilities": {
                "CRITICAL": 0.08,
                "HIGH": 0.72,
                "MODERATE": 0.15,
                "ROUTINE": 0.05,
            },
            "top_probability": 0.72,
        }
    """
    model = load_model()

    row = pd.DataFrame(
        [
            {
                "intake_message": intake_message,
                "risk_indicators": risk_indicators,
                "waiting_time_minutes": waiting_time_minutes,
            }
        ]
    )

    predicted_label = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = model.named_steps["clf"].classes_

    class_probabilities = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    top_probability = float(max(class_probabilities.values()))

    return {
        "predicted_label": str(predicted_label),
        "class_probabilities": class_probabilities,
        "top_probability": top_probability,
    }
