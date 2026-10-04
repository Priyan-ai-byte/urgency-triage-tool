"""
Threshold validation and FP/FN analysis for the uncertainty layer.

ANALYSIS ONLY. This script reads the held-out TEST SET, scores it with the
existing baseline model (through src/predict.py) and measures how different
confidence thresholds would behave. It never changes the production
thresholds (src/uncertainty.py), the model, the data, or any other project
file. It writes exactly two files:

    evaluation/threshold_results.csv
    evaluation/threshold_analysis.md

Question answered
-----------------
The production rule treats a model prediction as confident only when
    top_probability >= MIN_TOP_PROBABILITY  AND
    top_probability - second_probability >= MIN_MARGIN.
Otherwise the case is sent to NEEDS_CLARIFICATION (a human gathers more
information). For each (top, margin) pair in a grid this script reports how
many cases would be sent to clarification and how well the remaining model
decisions perform on urgent cases.

Definitions (urgent = CRITICAL + HIGH)
--------------------------------------
Only the model-confidence rule is simulated. Safety overrides, input-quality
flags and contradiction flags are NOT simulated; they can only add escalation
or uncertainty on top of what is measured here.

  retained        confident cases; the model decision is kept
  clarification   not confident; routed to a human, no automatic decision
  TP              urgent case, retained, predicted urgent
  FN              urgent case, retained, predicted NOT urgent  (silent miss)
  FP              non-urgent case, retained, predicted urgent
  TN              non-urgent case, retained, predicted NOT urgent
  urgent_recall               TP / (TP + FN)           (retained cases only)
  urgent_precision            TP / (TP + FP)           (retained cases only)
  false_positive_rate         FP / (FP + TN)           (retained cases only)
  urgent_recall_incl_review   (TP + urgent cases sent to clarification)
                              / all urgent cases
                              i.e. the share of urgent cases that were NOT
                              silently classified as non-urgent.

Running
-------
    python evaluation/threshold_analysis.py --test-csv data/test_split.csv

If --test-csv is omitted, data/test_split.csv (the saved 160-row held-out
split) is used when it exists.

Input schema (project defaults, each can be overridden from the command line)
-----------------------------------------------------------------------------
    intake_message          -> --message-col
    risk_indicators         -> --indicators-col
    waiting_time_minutes    -> --waiting-col
    urgency_label           -> --label-col

The existing pipeline is called through src.predict.predict_urgency(
intake_message, risk_indicators, waiting_time_minutes), which returns
{"predicted_label", "class_probabilities", "top_probability"}.

The project-specific part (where the held-out split lives and how the model
is called) is isolated in `load_cases`, `project_predict_fn` and
`normalize_prediction`. Everything else is pure and deterministic.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Read-only import: the CURRENT production thresholds, for reference only.
from src.uncertainty import MIN_MARGIN, MIN_TOP_PROBABILITY  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

TOP_PROBABILITY_GRID = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MARGIN_GRID = (0.05, 0.10, 0.15, 0.20)

SEVERITY_LABELS = ("ROUTINE", "MODERATE", "HIGH", "CRITICAL")
URGENT_LABELS = frozenset({"CRITICAL", "HIGH"})

# Workload cap used only to pick candidates (assumption, adjustable by CLI):
# sending nearly everything to a human trivially removes every silent miss, so
# candidates must keep the clarification rate at or below this value.
DEFAULT_MAX_CLARIFICATION_RATE = 0.40

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation"
CSV_NAME = "threshold_results.csv"
REPORT_NAME = "threshold_analysis.md"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"

DEFAULT_MESSAGE_COL = "intake_message"
DEFAULT_INDICATORS_COL = "risk_indicators"
DEFAULT_WAITING_COL = "waiting_time_minutes"
DEFAULT_LABEL_COL = "urgency_label"

# Older generic header for the message column. Used ONLY as a fallback when the
# default "intake_message" column is absent from the file.
LEGACY_MESSAGE_COL = "message"

DEFAULT_TEST_CSV_CANDIDATES = (
    "data/test_split.csv",
    "data/test.csv",
    "data/test_set.csv",
    "data/holdout_test.csv",
    "data/processed/test.csv",
    "data/processed/test_set.csv",
)

RESULT_COLUMNS = (
    "top_probability_threshold",
    "margin_threshold",
    "is_current_threshold",
    "n_total",
    "n_needs_clarification",
    "n_retained",
    "clarification_rate",
    "urgent_recall",
    "urgent_precision",
    "false_positives",
    "false_negatives",
    "false_positive_rate",
    "true_positives",
    "true_negatives",
    "urgent_cases_in_clarification",
    "non_urgent_cases_in_clarification",
    "urgent_recall_incl_review",
)


@dataclass(frozen=True)
class Case:
    """One scored test case."""

    true_label: str
    predicted_label: str
    probabilities: Mapping[str, float]


# --------------------------------------------------------------------------- #
# Pure analysis
# --------------------------------------------------------------------------- #

def threshold_grid(
    top_values: Sequence[float] = TOP_PROBABILITY_GRID,
    margin_values: Sequence[float] = MARGIN_GRID,
) -> List[Tuple[float, float]]:
    """All (top, margin) combinations, ordered by top then margin."""
    return list(itertools.product(top_values, margin_values))


def is_confident(
    probabilities: Mapping[str, float], min_top: float, min_margin: float
) -> bool:
    """
    Same rule as src.uncertainty.check_confidence, with variable thresholds:
    top >= min_top AND (top - second) >= min_margin. Values are rounded to 9
    decimals so float noise cannot flip boundary cases. Fewer than two
    probabilities means the confidence cannot be established, so not confident.
    """
    values = sorted((float(v) for v in probabilities.values()), reverse=True)
    if len(values) < 2:
        return False
    top = round(values[0], 9)
    margin = round(values[0] - values[1], 9)
    return top >= round(min_top, 9) and margin >= round(min_margin, 9)


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def evaluate_threshold(
    cases: Sequence[Case], min_top: float, min_margin: float
) -> Dict[str, Any]:
    """Metrics for one (top, margin) combination."""
    tp = fp = fn = tn = 0
    urgent_clarification = non_urgent_clarification = 0

    for case in cases:
        truly_urgent = case.true_label in URGENT_LABELS
        if not is_confident(case.probabilities, min_top, min_margin):
            if truly_urgent:
                urgent_clarification += 1
            else:
                non_urgent_clarification += 1
            continue
        predicted_urgent = case.predicted_label in URGENT_LABELS
        if truly_urgent and predicted_urgent:
            tp += 1
        elif truly_urgent:
            fn += 1
        elif predicted_urgent:
            fp += 1
        else:
            tn += 1

    n_total = len(cases)
    n_clarification = urgent_clarification + non_urgent_clarification
    total_urgent = tp + fn + urgent_clarification

    return {
        "top_probability_threshold": min_top,
        "margin_threshold": min_margin,
        "is_current_threshold": (
            abs(min_top - MIN_TOP_PROBABILITY) < 1e-9
            and abs(min_margin - MIN_MARGIN) < 1e-9
        ),
        "n_total": n_total,
        "n_needs_clarification": n_clarification,
        "n_retained": n_total - n_clarification,
        "clarification_rate": _ratio(n_clarification, n_total),
        "urgent_recall": _ratio(tp, tp + fn),
        "urgent_precision": _ratio(tp, tp + fp),
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": _ratio(fp, fp + tn),
        "true_positives": tp,
        "true_negatives": tn,
        "urgent_cases_in_clarification": urgent_clarification,
        "non_urgent_cases_in_clarification": non_urgent_clarification,
        "urgent_recall_incl_review": _ratio(tp + urgent_clarification, total_urgent),
    }


def evaluate_grid(
    cases: Sequence[Case],
    top_values: Sequence[float] = TOP_PROBABILITY_GRID,
    margin_values: Sequence[float] = MARGIN_GRID,
) -> List[Dict[str, Any]]:
    return [
        evaluate_threshold(cases, top, margin)
        for top, margin in threshold_grid(top_values, margin_values)
    ]


def select_candidates(
    rows: Sequence[Dict[str, Any]],
    max_clarification_rate: float = DEFAULT_MAX_CLARIFICATION_RATE,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rank candidates for urgent safety, NOT for accuracy.

    Eligible: clarification rate <= max_clarification_rate (workload cap).
    Order: fewest silent urgent misses (FN), then fewest false positives (FP),
    then fewest clarification cases, then the lower thresholds.
    """
    eligible = [
        r for r in rows
        if r["clarification_rate"] is not None
        and r["clarification_rate"] <= max_clarification_rate
    ]
    eligible.sort(
        key=lambda r: (
            r["false_negatives"],
            r["false_positives"],
            r["n_needs_clarification"],
            r["top_probability_threshold"],
            r["margin_threshold"],
        )
    )
    return eligible[:limit]


# --------------------------------------------------------------------------- #
# Output files
# --------------------------------------------------------------------------- #

def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: ("" if row[k] is None else row[k]) for k in RESULT_COLUMNS}
            )


def _num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Top prob. ≥ | Margin ≥ | Clarification | Retained | Urgent recall "
        "| Urgent precision | FP | FN | FPR | Urgent recall incl. review | Note |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        note = "current" if r["is_current_threshold"] else ""
        lines.append(
            f"| {r['top_probability_threshold']:.2f} | {r['margin_threshold']:.2f} "
            f"| {r['n_needs_clarification']} ({_num(r['clarification_rate'])}) "
            f"| {r['n_retained']} | {_num(r['urgent_recall'])} "
            f"| {_num(r['urgent_precision'])} | {r['false_positives']} "
            f"| {r['false_negatives']} | {_num(r['false_positive_rate'])} "
            f"| {_num(r['urgent_recall_incl_review'])} | {note} |"
        )
    return lines


def _recommendation(
    current: Optional[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    cap: float,
) -> List[str]:
    lines: List[str] = []
    if best is None:
        lines.append(
            f"No tested combination keeps the clarification rate at or below "
            f"{cap:.0%}. The workload cap or the model itself needs review "
            f"before thresholds can be compared."
        )
    elif current is None:
        lines.append(
            "The current thresholds are not part of the tested grid, so no "
            "direct comparison is possible."
        )
    else:
        same = (
            best["top_probability_threshold"] == current["top_probability_threshold"]
            and best["margin_threshold"] == current["margin_threshold"]
        )
        cur_key = (current["false_negatives"], current["false_positives"],
                   current["n_needs_clarification"])
        best_key = (best["false_negatives"], best["false_positives"],
                    best["n_needs_clarification"])
        if same or best_key >= cur_key:
            lines.append(
                "**Retain the current thresholds.** No tested combination within "
                "the workload cap does better on silent urgent misses, then on "
                "false positives, then on clarification workload."
            )
        elif best["false_negatives"] < current["false_negatives"]:
            lines.append(
                f"**Consider, but do not yet adopt,** top ≥ "
                f"{best['top_probability_threshold']:.2f} and margin ≥ "
                f"{best['margin_threshold']:.2f}. Compared with the current "
                f"thresholds it changes silent urgent misses from "
                f"{current['false_negatives']} to {best['false_negatives']}, "
                f"false positives from {current['false_positives']} to "
                f"{best['false_positives']}, and clarification cases from "
                f"{current['n_needs_clarification']} to "
                f"{best['n_needs_clarification']}."
            )
        else:
            lines.append(
                f"**Optional workload improvement only.** top ≥ "
                f"{best['top_probability_threshold']:.2f} and margin ≥ "
                f"{best['margin_threshold']:.2f} gives the same silent urgent "
                f"misses as the current thresholds with fewer clarification "
                f"cases ({best['n_needs_clarification']} vs "
                f"{current['n_needs_clarification']})."
            )
    lines.append("")
    lines.append(
        "This analysis changes no production threshold. Any change should be "
        "made deliberately, reviewed by a person, and first confirmed on a "
        "separate validation set, because the candidates above were chosen by "
        "looking at this test set and their numbers are therefore optimistic."
    )
    return lines


def build_report(
    rows: Sequence[Dict[str, Any]],
    reference: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    max_clarification_rate: float,
    source: str,
) -> str:
    n_total = rows[0]["n_total"] if rows else 0
    n_urgent = (
        reference["true_positives"]
        + reference["false_negatives"]
        + reference["urgent_cases_in_clarification"]
    )
    current = next((r for r in rows if r["is_current_threshold"]), None)
    strictest = max(rows, key=lambda r: (r["n_needs_clarification"],
                                          r["top_probability_threshold"],
                                          r["margin_threshold"]))
    loosest = min(rows, key=lambda r: (r["n_needs_clarification"],
                                        r["top_probability_threshold"],
                                        r["margin_threshold"]))

    out: List[str] = []
    out += [
        "# Threshold Analysis: Uncertainty Layer",
        "",
        "## Purpose",
        "",
        "The uncertainty layer sends a case to NEEDS_CLARIFICATION when the "
        "baseline model is not confident enough. This report measures, on the "
        "held-out test set, what different confidence thresholds would do to "
        "urgent-case recall, false positives, false negatives and the number of "
        "cases sent to a human. It is an analysis only: no production threshold "
        "was changed. The tool supports human review and does not diagnose.",
        "",
        "## Current thresholds",
        "",
        f"- `MIN_TOP_PROBABILITY = {MIN_TOP_PROBABILITY}`",
        f"- `MIN_MARGIN = {MIN_MARGIN}`",
        "",
        "A prediction is confidence-valid only when the top probability is at "
        "least `MIN_TOP_PROBABILITY` and the gap to the second-highest "
        "probability is at least `MIN_MARGIN`.",
        "",
        "## Threshold combinations tested",
        "",
        "- Top probability: " + ", ".join(f"{v:.2f}" for v in TOP_PROBABILITY_GRID),
        "- Probability margin: " + ", ".join(f"{v:.2f}" for v in MARGIN_GRID),
        f"- {len(rows)} combinations in total.",
        "",
        "## Data and method",
        "",
        f"- Source: {source}",
        f"- Test cases: {n_total}; truly urgent (CRITICAL + HIGH): {n_urgent}.",
        "- Urgent = CRITICAL + HIGH. Only the model-confidence rule is simulated; "
        "safety overrides, input-quality flags and contradiction flags are not "
        "(they can only add escalation or uncertainty).",
        "- A case below the thresholds is sent to clarification and receives no "
        "automatic decision. Metrics on retained cases use the model prediction.",
        "- **FN** = urgent case retained and predicted non-urgent (a silent miss). "
        "**FP** = non-urgent case retained and predicted urgent.",
        "- **Urgent recall** = TP / (TP + FN) over retained cases. "
        "**Urgent recall incl. review** = share of all urgent cases that were "
        "either predicted urgent or sent to a human, i.e. not silently "
        "classified as non-urgent.",
        "",
        "## Results",
        "",
    ]
    out += _table(rows)
    out += [
        "",
        "Reference without any clarification routing (every prediction retained): "
        f"urgent recall {_num(reference['urgent_recall'])}, urgent precision "
        f"{_num(reference['urgent_precision'])}, FP {reference['false_positives']}, "
        f"FN {reference['false_negatives']}, FPR "
        f"{_num(reference['false_positive_rate'])}.",
        "",
        "## Best-performing threshold candidates",
        "",
        "Candidates are ranked for urgent-case safety, not accuracy. Eligible "
        f"combinations keep the clarification rate at or below "
        f"{max_clarification_rate:.0%} (a workload assumption). They are ordered "
        "by fewest silent urgent misses (FN), then fewest false positives, then "
        "fewest clarification cases.",
        "",
    ]
    out += _table(candidates) if candidates else [
        "No combination met the clarification-rate cap."
    ]
    out += ["", "## Interpretation", ""]
    if current is not None:
        out.append(
            f"- At the current thresholds {current['n_needs_clarification']} of "
            f"{n_total} cases ({_num(current['clarification_rate'])}) go to "
            f"clarification, with {current['false_negatives']} silent urgent "
            f"misses and {current['false_positives']} false positives."
        )
    out += [
        f"- The loosest tested setting (top ≥ "
        f"{loosest['top_probability_threshold']:.2f}, margin ≥ "
        f"{loosest['margin_threshold']:.2f}) sends "
        f"{loosest['n_needs_clarification']} cases to clarification and leaves "
        f"{loosest['false_negatives']} silent urgent misses; the strictest "
        f"(top ≥ {strictest['top_probability_threshold']:.2f}, margin ≥ "
        f"{strictest['margin_threshold']:.2f}) sends "
        f"{strictest['n_needs_clarification']} and leaves "
        f"{strictest['false_negatives']}.",
        "- Stricter thresholds trade human workload for fewer automatic "
        "decisions. The cost is clarification volume, so the cap above matters.",
        "- No threshold is called optimal here. Overall accuracy was not used to "
        "rank candidates; urgent-case recall and controlled false positives were.",
        "- Differences of a few cases on a single test set are not statistically "
        "reliable; no confidence intervals were computed.",
        "",
        "## Recommendation",
        "",
    ]
    out += _recommendation(current, candidates[0] if candidates else None,
                           max_clarification_rate)
    out += [
        "",
        "## Limitations",
        "",
        "- One held-out test set; small urgent counts make every figure noisy.",
        "- Thresholds judged on the same data they are chosen from are optimistic.",
        "- Safety overrides are not simulated, so real silent misses can only be "
        "equal to or lower than the FN shown.",
        "",
    ]
    return "\n".join(out)


def run_analysis(
    cases: Sequence[Case],
    output_dir: Path,
    max_clarification_rate: float = DEFAULT_MAX_CLARIFICATION_RATE,
    source: str = "held-out test set",
) -> Dict[str, Any]:
    """Evaluate the grid and write threshold_results.csv and threshold_analysis.md."""
    if not cases:
        raise ValueError("No test cases to evaluate")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate_grid(cases)
    reference = evaluate_threshold(cases, 0.0, 0.0)
    candidates = select_candidates(rows, max_clarification_rate)

    csv_path = output_dir / CSV_NAME
    report_path = output_dir / REPORT_NAME
    write_csv(rows, csv_path)
    report_path.write_text(
        build_report(rows, reference, candidates, max_clarification_rate, source),
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "reference": reference,
        "candidates": candidates,
        "csv_path": csv_path,
        "report_path": report_path,
    }


# --------------------------------------------------------------------------- #
# Project adapter: held-out test set + existing prediction pipeline
# --------------------------------------------------------------------------- #

_PREDICT_CANDIDATES = ("predict_urgency", "predict_message", "predict_one", "predict")
_LABEL_KEYS = ("prediction", "label", "predicted_label", "urgency")
_PROB_KEYS = ("probabilities", "probs", "probability", "class_probabilities")


def normalize_prediction(result: Any) -> Tuple[str, Dict[str, float]]:
    """Return (LABEL, {LABEL: probability}) from the prediction function's result."""
    if isinstance(result, Mapping):
        label = next((result[k] for k in _LABEL_KEYS if k in result), None)
        probs = next((result[k] for k in _PROB_KEYS if k in result), None)
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        label, probs = result
    else:
        raise TypeError(f"Unsupported prediction result: {type(result).__name__}")
    if not isinstance(label, str) or label.strip().upper() not in SEVERITY_LABELS:
        raise ValueError(f"Unexpected predicted label {label!r}")
    if not isinstance(probs, Mapping) or len(probs) < 2:
        raise ValueError("The prediction must include a probability for each class")
    return (
        label.strip().upper(),
        {str(k).strip().upper(): float(v) for k, v in probs.items()},
    )


def project_predict_fn() -> Callable[..., Any]:
    """
    The EXISTING prediction pipeline in src/predict.py (adapter).

    Called as fn(intake_message, risk_indicators, waiting_time_minutes), which
    matches src.predict.predict_urgency. That function returns
    {"predicted_label", "class_probabilities", "top_probability"}, which
    normalize_prediction() reads.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    module = importlib.import_module("src.predict")
    for name in _PREDICT_CANDIDATES:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        f"None of {_PREDICT_CANDIDATES} found in src/predict.py. "
        f"Edit project_predict_fn() in evaluation/threshold_analysis.py."
    )


def _to_waiting(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _to_risk_indicators(value: Optional[str]) -> Optional[str]:
    """
    An empty cell means the field is MISSING and is passed as None, which is
    how the training pipeline sees it (pandas reads an empty cell as NaN).
    The literal text "none" is a real value (explicitly no indicator) and is
    passed through unchanged; it must never be confused with a missing field.
    """
    if value is None or value.strip() == "":
        return None
    return value


def load_cases(
    test_csv: Path,
    predict_fn: Callable[..., Any],
    message_col: str = DEFAULT_MESSAGE_COL,
    indicators_col: str = DEFAULT_INDICATORS_COL,
    waiting_col: str = DEFAULT_WAITING_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> List[Case]:
    """Score every row of the existing held-out test file. No re-splitting."""
    with open(test_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if (
            message_col == DEFAULT_MESSAGE_COL
            and message_col not in columns
            and LEGACY_MESSAGE_COL in columns
        ):
            message_col = LEGACY_MESSAGE_COL
        missing = [c for c in (message_col, indicators_col, waiting_col, label_col)
                   if c not in columns]
        if missing:
            raise ValueError(
                f"Column(s) {missing} not found in {test_csv}. "
                f"Available columns: {columns}. Use the --*-col options."
            )
        cases: List[Case] = []
        for row in reader:
            true_label = (row[label_col] or "").strip().upper()
            if true_label not in SEVERITY_LABELS:
                raise ValueError(
                    f"Unexpected true label {row[label_col]!r}; expected one of "
                    f"{SEVERITY_LABELS}. Check --label-col."
                )
            predicted, probabilities = normalize_prediction(
                predict_fn(
                    row[message_col] or "",
                    _to_risk_indicators(row[indicators_col]),
                    _to_waiting(row[waiting_col]),
                )
            )
            cases.append(Case(true_label, predicted, probabilities))
    return cases


def find_test_csv(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else Path.cwd() / path
    for candidate in DEFAULT_TEST_CSV_CANDIDATES:
        path = PROJECT_ROOT / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the held-out test file. Pass it with --test-csv "
        "(the same test split used for the baseline evaluation)."
    )


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze uncertainty thresholds on the held-out test set "
                    "(analysis only; no production file is changed)."
    )
    parser.add_argument("--test-csv",
                        help="existing held-out test split (CSV); "
                             "default: data/test_split.csv")
    parser.add_argument("--message-col", default=DEFAULT_MESSAGE_COL)
    parser.add_argument("--indicators-col", default=DEFAULT_INDICATORS_COL)
    parser.add_argument("--waiting-col", default=DEFAULT_WAITING_COL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--output-dir", default=None,
                        help="default: <project>/evaluation")
    parser.add_argument("--max-clarification-rate", type=float,
                        default=DEFAULT_MAX_CLARIFICATION_RATE,
                        help="workload cap used to pick candidates (default 0.40)")
    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    predict_fn: Optional[Callable[..., Any]] = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        test_csv = find_test_csv(args.test_csv)
        predict_fn = predict_fn or project_predict_fn()
        cases = load_cases(
            test_csv, predict_fn,
            args.message_col, args.indicators_col, args.waiting_col, args.label_col,
        )
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
        result = run_analysis(
            cases, output_dir, args.max_clarification_rate,
            source=f"`{test_csv.name}` (existing held-out test split)",
        )
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Evaluated {len(cases)} test cases, {len(result['rows'])} combinations.")
    print(f"Wrote {result['csv_path']}")
    print(f"Wrote {result['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
