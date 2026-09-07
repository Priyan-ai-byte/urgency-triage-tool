"""
features.py

Feature engineering for the Human-Reviewed Urgency Triage Tool baseline
model. Combines:

- TF-IDF text features from `intake_message` (unigrams + bigrams)
- Structured numeric/binary features derived from `risk_indicators` and
  `waiting_time_minutes`

All preprocessing that "learns" anything from data (the waiting-time
median used for imputation) lives inside scikit-learn transformers, so
when this pipeline is fit inside a Pipeline/train-test split, it is fit
ONLY on the training data - this is what prevents test-set information
from leaking into training.

IMPORTANT (target leakage): `professional_triage_outcome` and
`urgency_label` are targets, not features, and are never referenced by
anything in this module. Only `intake_message`, `risk_indicators`, and
`waiting_time_minutes` are used as model inputs.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

# Must stay in sync with the RISK_INDICATORS vocabulary used by
# data/generate_synthetic_data.py.
RISK_INDICATOR_CATEGORIES: List[str] = [
    "none",
    "distress",
    "hopelessness",
    "self_harm_mention",
    "immediate_safety_concern",
    "plan_indicator",
    "harm_to_others_indicator",
    "severe_distress",
]

# Model input columns. Deliberately excludes professional_triage_outcome
# and urgency_label, which are targets/ground truth only - never features.
FEATURE_COLUMNS: List[str] = ["intake_message", "risk_indicators", "waiting_time_minutes"]

TARGET_COLUMN: str = "urgency_label"

# Fixed class order used throughout the project.
CLASS_LABELS: List[str] = ["CRITICAL", "HIGH", "MODERATE", "ROUTINE"]

URGENT_LABELS = {"CRITICAL", "HIGH"}

RANDOM_STATE: int = 42


# --------------------------------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------------------------------

def fill_missing_text(messages: pd.Series) -> pd.Series:
    """
    Replace missing intake_message values with an empty string so
    TfidfVectorizer never receives a NaN.

    A plain named function (not a lambda) is used on purpose: lambdas
    cannot be pickled by joblib, which would silently break
    joblib.dump()/joblib.load() for the saved model pipeline.
    """
    return messages.fillna("")


def parse_risk_indicators(raw_value) -> List[str]:
    """
    Parse a raw risk_indicators cell into a list of lowercase, stripped
    indicator tokens.

    The field is parsed as a comma-separated list for robustness - the
    current dataset generator only ever writes a single indicator per
    row, but the project's input schema allows a list-like
    representation, so this handles that case defensively too.
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return []
    text = str(raw_value).strip()
    if text == "" or text.lower() == "nan":
        return []
    return [token.strip().lower() for token in text.split(",") if token.strip() != ""]


# --------------------------------------------------------------------------
# CUSTOM TRANSFORMER: risk indicators -> fixed binary feature columns
# --------------------------------------------------------------------------

class RiskIndicatorEncoder(BaseEstimator, TransformerMixin):
    """
    Converts the raw `risk_indicators` column into a fixed-width binary
    feature matrix:

    - One column per known category in RISK_INDICATOR_CATEGORIES (1.0 if
      that category is present in the row, else 0.0). Note "none" is
      itself a category here - it means a professional/form explicitly
      recorded "no risk indicator", which is different from the field
      being absent entirely.
    - One additional "risk_missing" column, set to 1.0 only when the raw
      field itself was missing (NaN) - a genuinely missing field is
      treated as "unknown", never silently treated as "no risk".

    This transformer is stateless (the category vocabulary is fixed
    ahead of time), so fit() does nothing but is required to satisfy the
    scikit-learn Transformer API.
    """

    def __init__(self, categories: Optional[List[str]] = None):
        # scikit-learn's clone() requires that whatever we store here under
        # the same name as the constructor argument is EXACTLY what was
        # passed in (including None) - any transformation (like resolving
        # the default here) breaks cloning inside a Pipeline. So we store
        # the raw argument and resolve the actual list lazily instead.
        self.categories = categories

    def _resolved_categories(self) -> List[str]:
        return list(self.categories) if self.categories is not None else list(RISK_INDICATOR_CATEGORIES)

    def fit(self, X, y=None):
        # This transformer is stateless (the category vocabulary is fixed
        # ahead of time), but scikit-learn's check_is_fitted (used
        # internally by Pipeline.transform) looks for at least one
        # trailing-underscore attribute to confirm fit() has been called.
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        categories = self._resolved_categories()
        values = np.asarray(X).reshape(-1)
        n_rows = len(values)
        n_categories = len(categories)
        category_index = {cat: i for i, cat in enumerate(categories)}

        # +1 column for the "field was missing entirely" flag.
        output = np.zeros((n_rows, n_categories + 1), dtype=float)

        for row_idx, raw_value in enumerate(values):
            is_missing = raw_value is None or (
                isinstance(raw_value, float) and np.isnan(raw_value)
            )
            if is_missing:
                output[row_idx, n_categories] = 1.0
                continue
            for token in parse_risk_indicators(raw_value):
                if token in category_index:
                    output[row_idx, category_index[token]] = 1.0

        return output

    def get_feature_names_out(self, input_features=None):
        return np.array([f"risk_{c}" for c in self._resolved_categories()] + ["risk_missing"])


# --------------------------------------------------------------------------
# PIPELINE ASSEMBLY
# --------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build the combined text + structured feature preprocessor.

    - intake_message -> fill missing text -> TF-IDF (unigrams + bigrams)
    - waiting_time_minutes -> median imputation (+ missing indicator) -> scaling
    - risk_indicators -> fixed binary category encoding (+ missing indicator)

    ColumnTransformer keeps the whole feature pipeline as a single,
    inspectable, joblib-picklable object - simpler to maintain than
    manually building and hstacking each piece by hand, while still
    keeping each of the three input columns' preprocessing independent
    and clearly separated.
    """
    text_pipeline = Pipeline(
        steps=[
            ("fill_missing", FunctionTransformer(fill_missing_text, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            # add_indicator=True appends a binary "was this missing" column
            # automatically, so missing waiting_time_minutes values are
            # handled safely without dropping any rows.
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )

    risk_pipeline = Pipeline(
        steps=[
            ("encode", RiskIndicatorEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            # A bare column name (not a list) makes ColumnTransformer pass
            # a 1D Series, which TfidfVectorizer requires.
            ("text", text_pipeline, "intake_message"),
            ("numeric", numeric_pipeline, ["waiting_time_minutes"]),
            ("risk", risk_pipeline, ["risk_indicators"]),
        ]
    )

    return preprocessor


def build_pipeline() -> Pipeline:
    """
    Build the full baseline pipeline: preprocessing + Logistic Regression.

    class_weight="balanced" compensates for the deliberately imbalanced
    class distribution (ROUTINE 40% / MODERATE 30% / HIGH 20% / CRITICAL
    10%) without needing manual resampling - an appropriate, low-effort
    choice for an interpretable baseline.
    """
    preprocessor = build_preprocessor()

    classifier = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
    )

    pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            ("clf", classifier),
        ]
    )

    return pipeline
