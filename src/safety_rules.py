"""
safety_rules.py

Safe Decision Layer — first coding step (SAFE_DECISION_LAYER_DESIGN.md,
Section 5). Implements ONLY the safety override rules:

    A. immediate_safety_concern -> floor CRITICAL
    B. plan_indicator           -> floor CRITICAL
    C. self_harm_mention        -> floor HIGH
    D. harm_to_others_indicator -> floor HIGH
    E. waiting-time escalation: MODERATE prediction + no stronger
       override already fired + waiting_time_minutes > 1440 -> floor HIGH

CORE SAFETY PRINCIPLE: this module may only ever RAISE the floor on a
label. It never lowers the model's prediction, and it never lowers
another override's floor. There is no code path in this file that can
produce an effective label less severe than the model's own prediction
(see highest_severity() and apply_safety_overrides() below, and
test_model_prediction_never_downgraded in the test suite).

This module is intentionally dependency-free (standard library only) —
no scikit-learn, no NLP, no external calls, no hidden thresholds beyond
the two named constants below. That is a deliberate design choice, not
an oversight: a safety-override layer should be the most boring, most
auditable part of the whole system, not the cleverest. In particular,
this module does NOT import src/features.py (which pulls in
scikit-learn) — it has its own small, self-contained risk_indicators
parser instead, so it can be read, tested, and audited in complete
isolation from the ML stack.

Does NOT import src/predict.py or the trained model. Does NOT implement
input-quality checks, contradiction checks, or the confidence rule —
those belong to src/uncertainty.py (not yet built). Does NOT implement
the final precedence logic that combines this module's output with
uncertainty — that belongs to src/triage.py (not yet built).
"""

from typing import List, Optional, Tuple

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

# Severity order, lowest to highest. Position in this list = severity rank.
SEVERITY_ORDER = ["ROUTINE", "MODERATE", "HIGH", "CRITICAL"]

# The only two "magic numbers" in this module — both named, both
# documented, both easy to find and change in one place if the design
# doc's thresholds are ever revised.
WAITING_TIME_ESCALATION_THRESHOLD_MINUTES = 1440  # 24 hours
WAITING_TIME_ESCALATION_FLOOR = "HIGH"

# Risk-indicator tags that trigger a direct override, and the floor each
# one applies (rules A-D). Which one "wins" when several are present is
# resolved separately by highest_severity() — order in this dict does
# not matter.
INDICATOR_OVERRIDES = {
    "immediate_safety_concern": "CRITICAL",
    "plan_indicator": "CRITICAL",
    "self_harm_mention": "HIGH",
    "harm_to_others_indicator": "HIGH",
}


# --------------------------------------------------------------------------
# SEVERITY HELPERS
# --------------------------------------------------------------------------

def _severity_rank(label: Optional[str]) -> int:
    """
    Map a label to its position in SEVERITY_ORDER (higher = more severe).
    Unknown or missing labels rank below ROUTINE (-1), so they never
    accidentally win a max() comparison against a real label.
    """
    if label is None:
        return -1
    try:
        return SEVERITY_ORDER.index(label)
    except ValueError:
        return -1


def highest_severity(*labels: Optional[str]) -> Optional[str]:
    """
    Return whichever of the given labels is most severe. None values are
    ignored unless every input is None, in which case None is returned.

    This is the single function that decides "which floor wins" whenever
    more than one override rule fires, and it is also what guarantees
    the model's prediction can never be downgraded: every call site in
    this module that computes a final label always includes
    model_prediction as one of the inputs to this function.
    """
    candidates = [label for label in labels if label is not None]
    if not candidates:
        return None
    return max(candidates, key=_severity_rank)


# --------------------------------------------------------------------------
# RISK INDICATOR PARSING (self-contained — see module docstring)
# --------------------------------------------------------------------------

def parse_risk_indicators(raw_value) -> List[str]:
    """
    Parse a raw risk_indicators cell into a list of lowercase, stripped
    tokens (comma-separated, so a single cell can hold more than one tag,
    e.g. "plan_indicator,self_harm_mention").

    Returns an empty list for None, NaN, an empty/whitespace-only string,
    or the literal value "none". This module has no override rule keyed
    on "none" or on a missing field, so all three cases are equivalent
    here: zero override-triggering tokens either way. (Distinguishing a
    genuinely missing field from an explicit "none" value matters
    elsewhere — that's an input-quality/uncertainty concern, handled
    later in src/uncertainty.py, not a safety-override concern here.)
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and raw_value != raw_value:  # NaN check, no pandas/numpy needed
        return []
    text = str(raw_value).strip().lower()
    if text in ("", "nan", "none"):
        return []
    return [token.strip() for token in text.split(",") if token.strip() != ""]


def _is_missing_waiting_time(value) -> bool:
    """True for None or NaN. Any real number (including 0) is NOT missing."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return False


# --------------------------------------------------------------------------
# INDIVIDUAL RULE CHECKS (rules A-D, then rule E)
# --------------------------------------------------------------------------

def check_indicator_overrides(risk_indicators) -> Tuple[Optional[str], List[str]]:
    """
    Apply rules A-D: check the raw risk_indicators field against the
    four direct indicator-based overrides.

    Returns (floor, matched_rules):
        floor          - the highest-severity floor triggered among any
                         matching tags, or None if no override-triggering
                         tag was present.
        matched_rules  - a list of human-readable strings, one per
                         matching tag, e.g. ["plan_indicator -> CRITICAL"].
                         If risk_indicators holds more than one
                         override-triggering tag (e.g.
                         "plan_indicator,self_harm_mention"), every match
                         is listed, but the returned floor is only ever
                         the single highest severity among them.
    """
    tokens = parse_risk_indicators(risk_indicators)
    floor: Optional[str] = None
    matched_rules: List[str] = []

    for token in tokens:
        if token in INDICATOR_OVERRIDES:
            rule_floor = INDICATOR_OVERRIDES[token]
            matched_rules.append(f"{token} -> {rule_floor}")
            floor = highest_severity(floor, rule_floor)

    return floor, matched_rules


def check_waiting_time_escalation(
    model_prediction: Optional[str],
    waiting_time_minutes,
    already_fired_floor: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Apply rule E: a MODERATE prediction that has waited more than
    WAITING_TIME_ESCALATION_THRESHOLD_MINUTES, with no stronger override
    already in play, is nudged up to HIGH.

    Guarded so this rule can NEVER:
      - apply to anything other than a MODERATE prediction — it is a
        targeted nudge for one specific situation, not a general
        wait-time rule that reinterprets every label.
      - produce anything other than HIGH — never CRITICAL. Waiting time
        alone says nothing about clinical risk content, and must never
        be read as evidence of imminent danger on its own.
      - fire on top of a stronger override that already set a floor of
        HIGH or CRITICAL from an indicator rule — a long wait is not
        more informative than an explicit structured risk indicator, so
        it should never appear to be "the reason" a case was escalated
        when a stronger signal was already present. (It is harmless to
        skip in this case anyway, since it could only ever propose HIGH,
        which would not raise an already-HIGH-or-higher floor further.)

    Returns (floor, reason): floor is "HIGH" or None; reason is a
    human-readable string, or None if the rule did not fire.
    """
    if model_prediction != "MODERATE":
        return None, None

    if _severity_rank(already_fired_floor) >= _severity_rank(WAITING_TIME_ESCALATION_FLOOR):
        return None, None

    if _is_missing_waiting_time(waiting_time_minutes):
        return None, None

    if waiting_time_minutes > WAITING_TIME_ESCALATION_THRESHOLD_MINUTES:
        reason = (
            f"waiting_time_minutes={waiting_time_minutes} > "
            f"{WAITING_TIME_ESCALATION_THRESHOLD_MINUTES} with MODERATE prediction "
            f"-> {WAITING_TIME_ESCALATION_FLOOR}"
        )
        return WAITING_TIME_ESCALATION_FLOOR, reason

    return None, None


# --------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# --------------------------------------------------------------------------

def apply_safety_overrides(
    model_prediction: str,
    risk_indicators=None,
    waiting_time_minutes=None,
) -> dict:
    """
    Run all safety override rules (A-E) against a raw intake record and
    the baseline model's own prediction, and return a structured result
    for the future triage orchestrator (src/triage.py, not yet built).

    Parameters
    ----------
    model_prediction : str
        The baseline model's raw predicted label — one of "CRITICAL",
        "HIGH", "MODERATE", "ROUTINE". Required: rule E needs it to
        apply its MODERATE-only guard, and the final effective label is
        always computed against it.
    risk_indicators :
        The raw risk_indicators field — a string (possibly
        comma-separated), None, or NaN. Handled safely in all three forms.
    waiting_time_minutes :
        The raw waiting_time_minutes field — an int, float, None, or NaN.
        Handled safely in all four forms.

    Returns
    -------
    dict with keys:
        safety_override_triggered : bool
            True if ANY rule fired, even if it did not end up changing
            the label (because the model already predicted something at
            least as severe). Keeping this visible regardless of whether
            it "won" matters for reviewer trust and audit — see design
            doc Section 5.
        safety_override_reason : str
            Every rule that fired, joined by "; ". Empty string if
            nothing fired.
        safety_override_floor : str or None
            The single highest-severity floor across every rule that
            fired. None if nothing fired.
        effective_label : str
            highest_severity(model_prediction, safety_override_floor) —
            the label after this layer. By construction this can never
            be less severe than model_prediction.

    Safety guarantee: effective_label's severity is always >=
    model_prediction's severity. There is no code path in this function
    that can lower it — effective_label is computed via highest_severity(),
    which always includes model_prediction as one of its inputs.
    """
    matched_reasons: List[str] = []
    floor: Optional[str] = None

    # Rules A-D: direct indicator overrides. Evaluated first so rule E's
    # guard condition can see whether a stronger floor already fired.
    indicator_floor, indicator_matches = check_indicator_overrides(risk_indicators)
    if indicator_floor is not None:
        floor = highest_severity(floor, indicator_floor)
        matched_reasons.extend(indicator_matches)

    # Rule E: limited waiting-time escalation.
    waiting_floor, waiting_reason = check_waiting_time_escalation(
        model_prediction=model_prediction,
        waiting_time_minutes=waiting_time_minutes,
        already_fired_floor=floor,
    )
    if waiting_floor is not None:
        floor = highest_severity(floor, waiting_floor)
        matched_reasons.append(waiting_reason)

    effective_label = highest_severity(model_prediction, floor)

    return {
        "safety_override_triggered": floor is not None,
        "safety_override_reason": "; ".join(matched_reasons),
        "safety_override_floor": floor,
        "effective_label": effective_label,
    }
