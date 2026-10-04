# Threshold Analysis: Uncertainty Layer

## Purpose

The uncertainty layer sends a case to NEEDS_CLARIFICATION when the baseline model is not confident enough. This report measures, on the held-out test set, what different confidence thresholds would do to urgent-case recall, false positives, false negatives and the number of cases sent to a human. It is an analysis only: no production threshold was changed. The tool supports human review and does not diagnose.

## Current thresholds

- `MIN_TOP_PROBABILITY = 0.55`
- `MIN_MARGIN = 0.1`

A prediction is confidence-valid only when the top probability is at least `MIN_TOP_PROBABILITY` and the gap to the second-highest probability is at least `MIN_MARGIN`.

## Threshold combinations tested

- Top probability: 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
- Probability margin: 0.05, 0.10, 0.15, 0.20
- 24 combinations in total.

## Data and method

- Source: `test_split.csv` (existing held-out test split)
- Test cases: 160; truly urgent (CRITICAL + HIGH): 48.
- Urgent = CRITICAL + HIGH. Only the model-confidence rule is simulated; safety overrides, input-quality flags and contradiction flags are not (they can only add escalation or uncertainty).
- A case below the thresholds is sent to clarification and receives no automatic decision. Metrics on retained cases use the model prediction.
- **FN** = urgent case retained and predicted non-urgent (a silent miss). **FP** = non-urgent case retained and predicted urgent.
- **Urgent recall** = TP / (TP + FN) over retained cases. **Urgent recall incl. review** = share of all urgent cases that were either predicted urgent or sent to a human, i.e. not silently classified as non-urgent.

## Results

| Top prob. ≥ | Margin ≥ | Clarification | Retained | Urgent recall | Urgent precision | FP | FN | FPR | Urgent recall incl. review | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 0.05 | 12 (0.075) | 148 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.45 | 0.10 | 14 (0.087) | 146 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.45 | 0.15 | 15 (0.094) | 145 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.45 | 0.20 | 16 (0.100) | 144 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.50 | 0.05 | 15 (0.094) | 145 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.50 | 0.10 | 15 (0.094) | 145 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.50 | 0.15 | 15 (0.094) | 145 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.50 | 0.20 | 16 (0.100) | 144 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.55 | 0.05 | 20 (0.125) | 140 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.55 | 0.10 | 20 (0.125) | 140 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 | current |
| 0.55 | 0.15 | 20 (0.125) | 140 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.55 | 0.20 | 20 (0.125) | 140 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.60 | 0.05 | 24 (0.150) | 136 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.60 | 0.10 | 24 (0.150) | 136 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.60 | 0.15 | 24 (0.150) | 136 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.60 | 0.20 | 24 (0.150) | 136 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.65 | 0.05 | 33 (0.206) | 127 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.65 | 0.10 | 33 (0.206) | 127 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.65 | 0.15 | 33 (0.206) | 127 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.65 | 0.20 | 33 (0.206) | 127 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.70 | 0.05 | 41 (0.256) | 119 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.70 | 0.10 | 41 (0.256) | 119 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.70 | 0.15 | 41 (0.256) | 119 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.70 | 0.20 | 41 (0.256) | 119 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |

Reference without any clarification routing (every prediction retained): urgent recall 0.979, urgent precision 0.940, FP 3, FN 1, FPR 0.027.

## Best-performing threshold candidates

Candidates are ranked for urgent-case safety, not accuracy. Eligible combinations keep the clarification rate at or below 40% (a workload assumption). They are ordered by fewest silent urgent misses (FN), then fewest false positives, then fewest clarification cases.

| Top prob. ≥ | Margin ≥ | Clarification | Retained | Urgent recall | Urgent precision | FP | FN | FPR | Urgent recall incl. review | Note |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.45 | 0.05 | 12 (0.075) | 148 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.45 | 0.10 | 14 (0.087) | 146 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.45 | 0.15 | 15 (0.094) | 145 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.50 | 0.05 | 15 (0.094) | 145 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |
| 0.50 | 0.10 | 15 (0.094) | 145 | 1.000 | 1.000 | 0 | 0 | 0.000 | 1.000 |  |

## Interpretation

- At the current thresholds 20 of 160 cases (0.125) go to clarification, with 0 silent urgent misses and 0 false positives.
- The loosest tested setting (top ≥ 0.45, margin ≥ 0.05) sends 12 cases to clarification and leaves 0 silent urgent misses; the strictest (top ≥ 0.70, margin ≥ 0.20) sends 41 and leaves 0.
- Stricter thresholds trade human workload for fewer automatic decisions. The cost is clarification volume, so the cap above matters.
- No threshold is called optimal here. Overall accuracy was not used to rank candidates; urgent-case recall and controlled false positives were.
- Differences of a few cases on a single test set are not statistically reliable; no confidence intervals were computed.

## Recommendation

**Optional workload improvement only.** top ≥ 0.45 and margin ≥ 0.05 gives the same silent urgent misses as the current thresholds with fewer clarification cases (12 vs 20).

This analysis changes no production threshold. Any change should be made deliberately, reviewed by a person, and first confirmed on a separate validation set, because the candidates above were chosen by looking at this test set and their numbers are therefore optimistic.

## Limitations

- One held-out test set; small urgent counts make every figure noisy.
- Thresholds judged on the same data they are chosen from are optimistic.
- Safety overrides are not simulated, so real silent misses can only be equal to or lower than the FN shown.
