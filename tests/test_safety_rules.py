"""
test_safety_rules.py

Unit tests for src/safety_rules.py — the safety override layer (Safe
Decision Layer, Section 5). Covers each override rule in isolation, the
"highest floor wins" combination behaviour, missing-input robustness,
and — most importantly — the never-downgrade safety guarantee.

Run from the project root:
    pytest tests/test_safety_rules.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.safety_rules import (  # noqa: E402
    WAITING_TIME_ESCALATION_THRESHOLD_MINUTES,
    apply_safety_overrides,
)


# --------------------------------------------------------------------------
# A-D: direct indicator overrides
# --------------------------------------------------------------------------

def test_immediate_safety_concern_floors_critical():
    result = apply_safety_overrides(
        model_prediction="ROUTINE",
        risk_indicators="immediate_safety_concern",
        waiting_time_minutes=30,
    )
    assert result["effective_label"] == "CRITICAL"
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "CRITICAL"
    assert "immediate_safety_concern" in result["safety_override_reason"]


def test_plan_indicator_floors_critical():
    result = apply_safety_overrides(
        model_prediction="MODERATE",
        risk_indicators="plan_indicator",
        waiting_time_minutes=10,
    )
    assert result["effective_label"] == "CRITICAL"
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "CRITICAL"
    assert "plan_indicator" in result["safety_override_reason"]


def test_self_harm_mention_floors_high():
    result = apply_safety_overrides(
        model_prediction="ROUTINE",
        risk_indicators="self_harm_mention",
        waiting_time_minutes=50,
    )
    assert result["effective_label"] == "HIGH"
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "HIGH"
    assert "self_harm_mention" in result["safety_override_reason"]


def test_harm_to_others_indicator_floors_high():
    result = apply_safety_overrides(
        model_prediction="MODERATE",
        risk_indicators="harm_to_others_indicator",
        waiting_time_minutes=50,
    )
    assert result["effective_label"] == "HIGH"
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "HIGH"
    assert "harm_to_others_indicator" in result["safety_override_reason"]


# --------------------------------------------------------------------------
# E: waiting-time escalation and its guards
# --------------------------------------------------------------------------

def test_moderate_with_long_wait_escalates_to_high():
    result = apply_safety_overrides(
        model_prediction="MODERATE",
        risk_indicators="distress",  # not an override-triggering tag
        waiting_time_minutes=WAITING_TIME_ESCALATION_THRESHOLD_MINUTES + 1,
    )
    assert result["effective_label"] == "HIGH"
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "HIGH"
    assert "waiting_time_minutes" in result["safety_override_reason"]


def test_waiting_time_at_or_below_threshold_does_not_escalate():
    # Exactly at the threshold: rule requires STRICTLY greater than 1440.
    result_at_threshold = apply_safety_overrides(
        model_prediction="MODERATE",
        risk_indicators="distress",
        waiting_time_minutes=WAITING_TIME_ESCALATION_THRESHOLD_MINUTES,
    )
    assert result_at_threshold["effective_label"] == "MODERATE"
    assert result_at_threshold["safety_override_triggered"] is False

    result_below_threshold = apply_safety_overrides(
        model_prediction="MODERATE",
        risk_indicators="distress",
        waiting_time_minutes=100,
    )
    assert result_below_threshold["effective_label"] == "MODERATE"
    assert result_below_threshold["safety_override_triggered"] is False


def test_waiting_time_rule_does_not_escalate_routine():
    result = apply_safety_overrides(
        model_prediction="ROUTINE",
        risk_indicators="none",
        waiting_time_minutes=WAITING_TIME_ESCALATION_THRESHOLD_MINUTES + 500,
    )
    # Rule E is MODERATE-only by design - a long-waiting ROUTINE case
    # must stay ROUTINE, never get nudged to HIGH.
    assert result["effective_label"] == "ROUTINE"
    assert result["safety_override_triggered"] is False


def test_waiting_time_rule_does_not_escalate_high():
    result = apply_safety_overrides(
        model_prediction="HIGH",
        risk_indicators="distress",
        waiting_time_minutes=WAITING_TIME_ESCALATION_THRESHOLD_MINUTES + 500,
    )
    # Rule E only applies to MODERATE predictions. A HIGH case with a
    # long wait should stay HIGH via this rule (no escalation attempted -
    # it has nothing to add, since HIGH is already its own ceiling).
    assert result["effective_label"] == "HIGH"
    assert result["safety_override_triggered"] is False


# --------------------------------------------------------------------------
# Combining multiple overrides
# --------------------------------------------------------------------------

def test_multiple_overrides_highest_floor_wins():
    # Both plan_indicator (CRITICAL) and self_harm_mention (HIGH) present
    # at once - CRITICAL must win.
    result = apply_safety_overrides(
        model_prediction="ROUTINE",
        risk_indicators="self_harm_mention,plan_indicator",
        waiting_time_minutes=10,
    )
    assert result["effective_label"] == "CRITICAL"
    assert result["safety_override_floor"] == "CRITICAL"
    assert "self_harm_mention" in result["safety_override_reason"]
    assert "plan_indicator" in result["safety_override_reason"]


def test_indicator_override_takes_precedence_over_waiting_time_reason():
    # MODERATE + harm_to_others_indicator (-> HIGH) + a long wait that
    # would ALSO independently suggest HIGH. The indicator rule should
    # fire; the waiting-time rule should not duplicate/override it since
    # an equal-or-stronger floor already fired.
    result = apply_safety_overrides(
        model_prediction="MODERATE",
        risk_indicators="harm_to_others_indicator",
        waiting_time_minutes=WAITING_TIME_ESCALATION_THRESHOLD_MINUTES + 100,
    )
    assert result["effective_label"] == "HIGH"
    assert "harm_to_others_indicator" in result["safety_override_reason"]
    assert "waiting_time_minutes" not in result["safety_override_reason"]


# --------------------------------------------------------------------------
# Missing / malformed input robustness
# --------------------------------------------------------------------------

def test_missing_risk_indicator_does_not_crash():
    for missing_value in (None, float("nan"), "", "none", "NaN"):
        result = apply_safety_overrides(
            model_prediction="MODERATE",
            risk_indicators=missing_value,
            waiting_time_minutes=100,
        )
        assert result["effective_label"] == "MODERATE"
        assert result["safety_override_triggered"] is False


def test_missing_waiting_time_does_not_crash():
    for missing_value in (None, float("nan")):
        result = apply_safety_overrides(
            model_prediction="MODERATE",
            risk_indicators="distress",
            waiting_time_minutes=missing_value,
        )
        # No crash, and the waiting-time rule simply cannot fire without
        # a value to compare against the threshold.
        assert result["effective_label"] == "MODERATE"
        assert result["safety_override_triggered"] is False

    # Missing waiting time must not interfere with an indicator override
    # firing normally.
    result_with_indicator = apply_safety_overrides(
        model_prediction="ROUTINE",
        risk_indicators="plan_indicator",
        waiting_time_minutes=None,
    )
    assert result_with_indicator["effective_label"] == "CRITICAL"


def test_missing_model_prediction_inputs_handled_gracefully():
    # Defensive check: even with nothing present anywhere, the function
    # returns a well-formed result rather than raising.
    result = apply_safety_overrides(
        model_prediction="ROUTINE",
        risk_indicators=None,
        waiting_time_minutes=None,
    )
    assert result["effective_label"] == "ROUTINE"
    assert result["safety_override_triggered"] is False
    assert result["safety_override_reason"] == ""
    assert result["safety_override_floor"] is None


# --------------------------------------------------------------------------
# The core safety guarantee: never downgrade the model's prediction
# --------------------------------------------------------------------------

def test_model_prediction_is_never_downgraded():
    severity_order = ["ROUTINE", "MODERATE", "HIGH", "CRITICAL"]

    # Case 1: no override fires at all - effective label must equal the
    # model's own prediction, for every possible prediction.
    for label in severity_order:
        result = apply_safety_overrides(
            model_prediction=label,
            risk_indicators="none",
            waiting_time_minutes=30,
        )
        assert result["effective_label"] == label

    # Case 2: an override rule fires with a floor LOWER than what the
    # model already predicted (e.g. model says CRITICAL, but the only
    # matching indicator rule would floor at HIGH). The model's own
    # higher prediction must win - this is the actual "never downgrade"
    # scenario, as opposed to merely "no override fired".
    result = apply_safety_overrides(
        model_prediction="CRITICAL",
        risk_indicators="self_harm_mention",  # would floor at HIGH alone
        waiting_time_minutes=30,
    )
    assert result["effective_label"] == "CRITICAL"
    # The override still fired and is still reported, just didn't change
    # the final label - visible for audit purposes (design doc Section 5).
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "HIGH"

    # Case 3: same idea, with the waiting-time rule. It can only ever
    # propose HIGH, so a model prediction of CRITICAL must never be
    # pulled down to HIGH by it. (The rule's own MODERATE-only guard
    # already prevents it from firing here at all, which is itself part
    # of the guarantee.)
    result = apply_safety_overrides(
        model_prediction="CRITICAL",
        risk_indicators="none",
        waiting_time_minutes=WAITING_TIME_ESCALATION_THRESHOLD_MINUTES + 1000,
    )
    assert result["effective_label"] == "CRITICAL"
    assert result["safety_override_triggered"] is False
