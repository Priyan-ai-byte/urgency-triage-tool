"""
End-to-end edge / failure-case evaluation for the integrated triage system.

Every test calls the public API `src.triage.triage()`. The baseline model is
replaced by a tiny deterministic fake (so no trained model is needed and the
model's label and confidence are fully controlled). The safety rules are NOT
faked: `triage()` calls the real `src.safety_rules.apply_safety_overrides`
through its default adapter, and the real uncertainty layer runs unchanged.
Nothing here re-implements production logic; the tests only state the expected
outcome for each failure mode.

Decision rules being protected (see src/triage.py):
  * effective label = highest of (model prediction, safety floor)
  * safety override present  -> final = effective label (wins over uncertainty)
  * CRITICAL is never replaced by NEEDS_CLARIFICATION
  * no override + uncertainty -> NEEDS_CLARIFICATION
"""

import pytest

from src.triage import NEEDS_CLARIFICATION, SEVERITY_ORDER, triage

# --------------------------------------------------------------------------- #
# Shared inputs and helpers
# --------------------------------------------------------------------------- #

CLEAN_MESSAGE = "I have had a mild cough and sore throat since yesterday"
NO_INDICATORS = "none"      # a VALID answer, different from "missing"
CLEAN_WAITING = 20

# Weak model output: top probability < 0.55 and a small margin.
LOW_CONFIDENCE = {"ROUTINE": 0.35, "MODERATE": 0.33, "HIGH": 0.20, "CRITICAL": 0.12}
# Top probability is fine but the runner-up is too close (margin < 0.10).
LOW_MARGIN = {"ROUTINE": 0.58, "MODERATE": 0.52, "HIGH": 0.05, "CRITICAL": 0.02}


def indicators(*names):
    """Structured risk indicators as a raw comma-separated string (the format
    the real safety_rules parser expects), e.g. "plan_indicator,self_harm_mention"."""
    return ",".join(names)


def confident(label):
    """Probabilities with a clear winner (top 0.70, margin 0.55)."""
    others = [lbl for lbl in SEVERITY_ORDER if lbl != label]
    return {label: 0.70, others[0]: 0.15, others[1]: 0.10, others[2]: 0.05}


def model(label, probs=None):
    """Deterministic stand-in for the baseline model."""
    p = probs if probs is not None else confident(label)
    return lambda message, risk_indicators, waiting_time: {
        "prediction": label,
        "probabilities": dict(p),
    }


def strict_model(label):
    """Like a real model that cannot score an empty message."""
    def predict(message, risk_indicators, waiting_time):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("cannot score an empty message")
        return {"prediction": label, "probabilities": confident(label)}
    return predict


def run(message=CLEAN_MESSAGE, risk_indicators=NO_INDICATORS, waiting=CLEAN_WAITING,
        label="ROUTINE", probs=None):
    return triage(message, risk_indicators, waiting, predict_fn=model(label, probs))


def evidence_text(result):
    return " ".join(result["evidence"])


def assert_reviewable(result):
    """Every outcome must be explainable and routed to a human."""
    assert result["evidence"]
    assert all(isinstance(line, str) and line for line in result["evidence"])
    assert "does not diagnose" in evidence_text(result)
    assert result["human_review"]["required"] is True


# --------------------------------------------------------------------------- #
# 1. Missing intake message
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("message", [None, "", "   "])
def test_missing_message_does_not_crash_and_asks_for_clarification(message):
    # Failure mode: a model that cannot score an empty message crashes the
    # pipeline, or the case silently disappears.
    result = triage(
        message, NO_INDICATORS, CLEAN_WAITING, predict_fn=strict_model("ROUTINE")
    )
    assert "message_missing" in result["input_quality_flags"]
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["final_recommendation"] == NEEDS_CLARIFICATION
    assert "baseline_model: ValueError" in result["component_errors"]
    assert result["model_prediction"] is None
    assert_reviewable(result)


@pytest.mark.parametrize("message", [None, "", "   "])
def test_missing_message_is_never_silently_routine(message):
    # Failure mode: a model that still returns ROUTINE for a blank message
    # lets an empty intake be reported as a routine case.
    result = run(message=message, label="ROUTINE")
    assert result["model_prediction"] == "ROUTINE"
    assert result["final_recommendation"] == NEEDS_CLARIFICATION
    assert result["final_recommendation"] != "ROUTINE"


def test_missing_message_does_not_block_a_safety_override():
    # Failure mode: incomplete text must not switch off an indicator-based
    # safety escalation.
    result = run(message=None, risk_indicators=indicators("plan_indicator"),
                 label="MODERATE")
    assert "message_missing" in result["input_quality_flags"]
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "CRITICAL"


# --------------------------------------------------------------------------- #
# 2. Very short intake message (fewer than 4 whitespace-separated tokens)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("message", ["help", "chest pain", "I need help"])
def test_very_short_message_is_insufficient_not_routine(message):
    # Failure mode: one to three words are classified as ROUTINE with
    # apparent confidence instead of being treated as insufficient input.
    result = run(message=message, label="ROUTINE")
    assert len(message.split()) < 4
    assert "message_too_short" in result["input_quality_flags"]
    assert result["safety_override_triggered"] is False
    assert result["final_recommendation"] == NEEDS_CLARIFICATION
    assert result["final_recommendation"] != "ROUTINE"
    assert_reviewable(result)


def test_four_token_message_is_not_flagged_as_too_short():
    # Boundary: exactly 4 tokens is long enough (low-content is a separate check).
    result = run(message="I need urgent help", label="MODERATE")
    assert "message_too_short" not in result["input_quality_flags"]


def test_short_message_does_not_block_a_safety_override():
    # Failure mode: "insufficient input" must not hide a genuine escalation.
    result = run(message="help", risk_indicators=indicators("plan_indicator"),
                 label="MODERATE")
    assert "message_too_short" in result["input_quality_flags"]
    assert result["final_recommendation"] == "CRITICAL"


# --------------------------------------------------------------------------- #
# 3. High-risk message contradicted by the structured risk indicator
# --------------------------------------------------------------------------- #

SUICIDAL_MESSAGE = (
    "I am thinking about suicide and I do not see a reason to go on"
)


def test_high_risk_text_with_none_indicator_is_flagged_and_not_downgraded():
    # Failure mode: indicators say "none", so strong self-harm wording is
    # trusted away. A CRITICAL model prediction must stay CRITICAL and the
    # contradiction must be recorded for the reviewer.
    result = run(message=SUICIDAL_MESSAGE, risk_indicators=NO_INDICATORS,
                 label="CRITICAL")
    assert "strong_text_no_indicator" in result["contradiction_flags"]
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["final_recommendation"] == "CRITICAL"
    assert "strong risk wording" in evidence_text(result)


def test_high_risk_text_with_none_indicator_is_never_routine():
    # Failure mode: the model misses the wording AND the indicator says
    # "none", so the case comes out ROUTINE. It must go to a human instead.
    result = run(message=SUICIDAL_MESSAGE, risk_indicators=NO_INDICATORS,
                 label="ROUTINE")
    assert "strong_text_no_indicator" in result["contradiction_flags"]
    assert result["final_recommendation"] != "ROUTINE"
    assert result["final_recommendation"] == NEEDS_CLARIFICATION
    assert_reviewable(result)


def test_reassuring_text_does_not_cancel_a_risk_indicator_override():
    # Failure mode: "I feel fine" wording contradicts a structured plan
    # indicator and is used to cancel the escalation. The mismatch is only
    # recorded; the safety override still wins.
    result = run(
        message="Just checking, I am feeling fine and have no symptoms today",
        risk_indicators=indicators("plan_indicator"),
        label="ROUTINE",
    )
    assert "structured_indicator_text_mismatch" in result["contradiction_flags"]
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "CRITICAL"


# --------------------------------------------------------------------------- #
# 4. Safety override with low model confidence
# --------------------------------------------------------------------------- #

def test_plan_indicator_escalates_despite_low_top_probability():
    # Failure mode: a weak model prediction routes an explicit plan /
    # immediate-risk case to "clarification" instead of escalating.
    result = run(
        message="I have a plan to end my life tonight and the means are ready",
        risk_indicators=indicators("plan_indicator"),
        label="MODERATE",
        probs=LOW_CONFIDENCE,
    )
    assert result["uncertainty"]["confidence_ok"] is False
    assert "low_top_probability" in result["uncertainty"]["confidence_reasons"]
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "CRITICAL"
    assert result["final_recommendation"] == "CRITICAL"
    assert "plan_indicator" in evidence_text(result)


def test_self_harm_mention_escalates_despite_low_margin():
    # Failure mode: a narrow probability margin suppresses a HIGH floor.
    result = run(
        message="I have been feeling very low and keep thinking about hurting myself",
        risk_indicators=indicators("self_harm_mention"),
        label="ROUTINE",
        probs=LOW_MARGIN,
    )
    assert "low_probability_margin" in result["uncertainty"]["confidence_reasons"]
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "HIGH"


# --------------------------------------------------------------------------- #
# 5. Moderate prediction with very long waiting time (> 1440 minutes)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("waiting", [1441, 1500, 4000])
def test_moderate_with_waiting_over_24_hours_escalates_to_high(waiting):
    # Failure mode: a patient left waiting for more than a day stays MODERATE.
    result = run(waiting=waiting, label="MODERATE")
    assert result["uncertainty"]["needs_clarification"] is False
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "HIGH"
    assert result["final_recommendation"] == "HIGH"
    assert "raised the level from MODERATE to HIGH" in evidence_text(result)
    assert f"Reported waiting time: {waiting} minutes." in evidence_text(result)


def test_moderate_with_short_waiting_is_not_escalated():
    # Control for the case above: no escalation without the long wait.
    result = run(waiting=30, label="MODERATE")
    assert result["safety_override_triggered"] is False
    assert result["final_recommendation"] == "MODERATE"


@pytest.mark.parametrize("label", ["HIGH", "CRITICAL"])
def test_long_waiting_never_lowers_a_higher_prediction(label):
    # Failure mode: a waiting-time floor overwrites a higher model label.
    result = run(waiting=3000, label=label)
    assert result["final_recommendation"] == label


# --------------------------------------------------------------------------- #
# 6. Ambiguous / low-confidence input with no safety override
# --------------------------------------------------------------------------- #

AMBIGUOUS_MESSAGE = "I have had some stomach cramps and feel a bit unwell since this morning"


@pytest.mark.parametrize(
    "probs, expected_reason",
    [
        (LOW_CONFIDENCE, "low_top_probability"),
        (LOW_MARGIN, "low_probability_margin"),
    ],
)
def test_low_confidence_without_override_needs_clarification(probs, expected_reason):
    # Failure mode: a weak prediction is reported as if it were reliable.
    result = run(message=AMBIGUOUS_MESSAGE, label="MODERATE", probs=probs)
    assert result["safety_override_triggered"] is False
    assert expected_reason in result["uncertainty"]["confidence_reasons"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION
    assert "more information is needed" in evidence_text(result)
    assert_reviewable(result)


def test_confident_version_of_same_input_keeps_the_model_prediction():
    # Control: the same input with a clear model result is not sent to clarification.
    result = run(message=AMBIGUOUS_MESSAGE, label="MODERATE")
    assert result["uncertainty"]["needs_clarification"] is False
    assert result["final_recommendation"] == "MODERATE"


# --------------------------------------------------------------------------- #
# 7. Historical-risk wording
# --------------------------------------------------------------------------- #

def test_historical_reference_adds_uncertainty_without_override():
    # Failure mode: a past event ("a few months ago") is treated as current,
    # or silently ignored. It must add uncertainty for a human to resolve.
    result = run(
        message="I had a bad cough a few months ago and it came back this week",
        label="ROUTINE",
    )
    assert result["contradiction_flags"] == ["possible_historical_reference"]
    assert result["safety_override_triggered"] is False
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_historical_reference_does_not_suppress_a_self_harm_override():
    # Failure mode: "years ago" wording is used to cancel a genuine current
    # safety override. The flag is added; the escalation stays.
    result = run(
        message="Years ago I hurt myself and lately I keep thinking about it again",
        risk_indicators=indicators("self_harm_mention"),
        label="ROUTINE",
    )
    assert "possible_historical_reference" in result["contradiction_flags"]
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "HIGH"


def test_historical_reference_does_not_suppress_a_plan_override():
    # Same protection for the highest-severity rule.
    result = run(
        message="I attempted suicide years ago and now I have a plan to do it again tonight",
        risk_indicators=indicators("plan_indicator"),
        label="MODERATE",
    )
    assert "possible_historical_reference" in result["contradiction_flags"]
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "CRITICAL"


# --------------------------------------------------------------------------- #
# 8. CRITICAL model prediction with uncertainty
# --------------------------------------------------------------------------- #

CRITICAL_WITH_UNCERTAINTY = [
    pytest.param(dict(waiting=None), id="missing_waiting_time"),
    pytest.param(dict(risk_indicators=None), id="missing_risk_indicators"),
    pytest.param(dict(message="help"), id="very_short_message"),
    pytest.param(dict(message=None), id="missing_message"),
    pytest.param(dict(probs=LOW_CONFIDENCE), id="low_confidence"),
    pytest.param(dict(probs=LOW_MARGIN), id="low_margin"),
    pytest.param(
        dict(message=SUICIDAL_MESSAGE, risk_indicators=NO_INDICATORS),
        id="strong_text_no_indicator",
    ),
    pytest.param(
        dict(message="I had chest pain years ago and it is back now",
             risk_indicators=NO_INDICATORS),
        id="historical_reference",
    ),
]


@pytest.mark.parametrize("overrides", CRITICAL_WITH_UNCERTAINTY)
def test_critical_is_never_downgraded_to_needs_clarification(overrides):
    # Failure mode: uncertainty of any kind turns a CRITICAL prediction into
    # NEEDS_CLARIFICATION, delaying the most urgent cases.
    result = run(label="CRITICAL", **overrides)
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["final_recommendation"] == "CRITICAL"
    assert result["final_recommendation"] != NEEDS_CLARIFICATION
    assert_reviewable(result)


def test_critical_with_uncertainty_tells_the_reviewer_why():
    result = run(label="CRITICAL", waiting=None)
    assert "never lowered" in evidence_text(result)
    assert "Waiting time was not provided" in evidence_text(result)


# --------------------------------------------------------------------------- #
# Cross-cutting guarantees
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label", ["ROUTINE", "MODERATE", "HIGH", "CRITICAL"])
def test_final_recommendation_is_never_below_the_model_prediction(label):
    # Failure mode: any layer lowers a model label. NEEDS_CLARIFICATION is the
    # only allowed non-severity outcome, and never when CRITICAL.
    for overrides in (
        {},
        {"waiting": None},
        {"probs": LOW_CONFIDENCE},
        {"message": "help"},
        {"waiting": 3000},
    ):
        result = run(label=label, **overrides)
        final = result["final_recommendation"]
        if final == NEEDS_CLARIFICATION:
            assert label != "CRITICAL"
        else:
            assert SEVERITY_ORDER.index(final) >= SEVERITY_ORDER.index(label)


def test_edge_case_outputs_are_deterministic():
    kwargs = dict(
        message="Years ago I hurt myself and lately I keep thinking about it again",
        risk_indicators=indicators("self_harm_mention"),
        waiting=None,
        label="MODERATE",
        probs=LOW_CONFIDENCE,
    )
    assert run(**kwargs) == run(**kwargs)
