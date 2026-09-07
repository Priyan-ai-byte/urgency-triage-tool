# Baseline Model Results (TF-IDF + Logistic Regression)

_Generated automatically by `training/train_baseline.py` on 2026-09-07 08:15 UTC, from actual measured output - no values below are invented._

## Dataset Information

- Source: `data/synthetic_intake_dataset.csv`
- Total rows: 800
- Class distribution (full dataset): CRITICAL=80, HIGH=160, MODERATE=240, ROUTINE=320

## Train/Test Split

- Split type: stratified
- Test size: 20%
- Random state: 42
- Training rows: 640
- Test rows: 160

## Model Description

- Algorithm: Logistic Regression (`class_weight='balanced'`, multinomial)
- Random state: 42
- This is an intentionally simple, interpretable baseline. It has not been
  tuned to hit any target metric - these are honest first-pass numbers, not
  optimised results.

## Feature Description

- **Text**: TF-IDF over `intake_message` (unigrams + bigrams, English stopwords, min_df=2)
- **Structured**: fixed binary encoding of `risk_indicators` (8 known categories + a
  separate "field was missing" flag), plus `waiting_time_minutes` (median-imputed on
  training data only, with its own missing-value flag, then scaled)
- `professional_triage_outcome` and `urgency_label` are never used as inputs (target leakage check)

## Overall Metrics

- **Accuracy**: 0.950
- **Macro F1**: 0.942
- **Weighted F1**: 0.950

## Per-Class Precision / Recall / F1

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| CRITICAL | 0.842 | 1.000 | 0.914 | 16 |
| HIGH | 0.935 | 0.906 | 0.921 | 32 |
| MODERATE | 0.960 | 1.000 | 0.980 | 48 |
| ROUTINE | 0.983 | 0.922 | 0.952 | 64 |

## Full Classification Report (scikit-learn text output)

```
              precision    recall  f1-score   support

    CRITICAL       0.84      1.00      0.91        16
        HIGH       0.94      0.91      0.92        32
    MODERATE       0.96      1.00      0.98        48
     ROUTINE       0.98      0.92      0.95        64

    accuracy                           0.95       160
   macro avg       0.93      0.96      0.94       160
weighted avg       0.95      0.95      0.95       160
```

## Confusion Matrix

Rows = true label, columns = predicted label.

```
                CRITICAL        HIGH    MODERATE     ROUTINE
CRITICAL              16           0           0           0
HIGH                   2          29           0           1
MODERATE               0           0          48           0
ROUTINE                1           2           2          59
```

## Urgent vs Not-Urgent Analysis (Analysis Only)

`URGENT` = CRITICAL + HIGH, `NOT_URGENT` = MODERATE + ROUTINE. This collapses
the model's normal 4-class predictions after the fact - no threshold tuning
has been applied at this stage, and the model was never trained on this binary
target.

- True positives (urgent correctly flagged): 47
- False negatives (urgent missed): 1
- False positives (not-urgent flagged as urgent): 3
- True negatives (not-urgent correctly left alone): 109
- **Urgent recall**: 0.979
- **Urgent precision**: 0.940
- **Urgent false-positive rate**: 0.027

## Interpretation

This is an unoptimised first baseline, measured honestly before any tuning. It has not yet been compared against the target metrics from the project specification, and no confidence threshold or safety-override logic has been applied - both belong to a later evaluation stage. False positives and false negatives should be examined qualitatively (see `training/test_prediction.py` output and a future dedicated false-positive/false-negative analysis) before drawing any conclusions about real-world readiness.
