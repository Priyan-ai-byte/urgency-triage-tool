"""
Tests for the triage integration layer (src/triage.py).

The baseline model and the safety rules are replaced by small fakes, so these
tests verify the integration logic only and do not need the trained model.

Safety contract used throughout: the safety function is called as
    apply_safety_overrides(model_prediction, risk_indicators, waiting_time_minutes)
i.e. with the baseline MODEL PREDICTION first, never the raw message.
"""

import ast
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from src import triage as triage_module
from src.triage import NEEDS_CLARIFICATION, SEVERITY_ORDER, max_severity, triage

CLEAN_MESSAGE = "I have had a mild cough and sore throat since yesterday"
CLEAN_INDICATORS = "none"
CLEAN_WAITING = 20

LOW_CONFIDENCE = {"ROUTINE": 0.40, "MODERATE": 0.35, "HIGH": 0.15, "CRITICAL": 0.10}
LOW_MARGIN = {"ROUTINE": 0.58, "MODERATE": 0.52, "HIGH": 0.05, "CRITICAL": 0.02}


# --------------------------------------------------------------------------- #
# Fakes and helpers
# --------------------------------------------------------------------------- #

def confident(label):
    others = [lbl for lbl in SEVERITY_ORDER if lbl != label]
    return {label: 0.70, others[0]: 0.15, others[1]: 0.10, others[2]: 0.05}


def fake_model(label, probs=None):
    p = probs if probs is not None else confident(label)
    return lambda message, risk_indicators, waiting_time: {
        "prediction": label,
        "probabilities": dict(p),
    }


def no_safety(model_prediction, risk_indicators, waiting_time):
    return {"triggered": False, "floor": None, "reason": None}


def fake_safety(floor, reason="test_rule"):
    return lambda model_prediction, risk_indicators, waiting_time: {
        "triggered": True,
        "floor": floor,
        "reason": reason,
    }


def run(message=CLEAN_MESSAGE, indicators=CLEAN_INDICATORS, waiting=CLEAN_WAITING,
        label="ROUTINE", probs=None, safety_fn=no_safety):
    return triage(
        message, indicators, waiting,
        predict_fn=fake_model(label, probs), safety_fn=safety_fn,
    )


def evidence_text(result):
    return " ".join(result["evidence"])


# --------------------------------------------------------------------------- #
# Severity helpers
# --------------------------------------------------------------------------- #

def test_severity_order():
    assert SEVERITY_ORDER == ("ROUTINE", "MODERATE", "HIGH", "CRITICAL")


@pytest.mark.parametrize(
    "labels, expected",
    [
        (("ROUTINE", "MODERATE"), "MODERATE"),
        (("HIGH", "MODERATE"), "HIGH"),
        (("CRITICAL", "HIGH"), "CRITICAL"),
        (("ROUTINE", None), "ROUTINE"),
        ((None, None), None),
    ],
)
def test_max_severity(labels, expected):
    assert max_severity(*labels) == expected


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        max_severity("URGENT")


# --------------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------------- #

def test_output_schema():
    result = run()
    required = {
        "model_prediction", "model_probabilities", "model_top_probability",
        "model_margin", "safety_override_triggered", "safety_override_reason",
        "safety_override_floor", "input_quality_flags", "contradiction_flags",
        "final_recommendation", "uncertainty", "evidence",
    }
    assert required <= set(result)
    assert isinstance(result["model_probabilities"], dict)
    assert isinstance(result["input_quality_flags"], list)
    assert isinstance(result["contradiction_flags"], list)
    assert isinstance(result["evidence"], list)
    assert isinstance(result["safety_override_triggered"], bool)
    assert {"needs_clarification", "reasons"} <= set(result["uncertainty"])
    assert abs(result["model_top_probability"] - 0.70) < 1e-9
    assert abs(result["model_margin"] - 0.55) < 1e-9


def test_components_receive_the_correct_arguments():
    calls = {}

    def spy_model(*args):
        calls["model"] = args
        return {"prediction": "MODERATE", "probabilities": confident("MODERATE")}

    def spy_safety(*args):
        calls["safety"] = args
        return None

    triage(CLEAN_MESSAGE, ["chest_pain"], 33, predict_fn=spy_model, safety_fn=spy_safety)
    # The model gets the raw input; safety gets the MODEL PREDICTION first.
    assert calls["model"] == (CLEAN_MESSAGE, ["chest_pain"], 33)
    assert calls["safety"] == ("MODERATE", ["chest_pain"], 33)
    assert calls["safety"][0] != CLEAN_MESSAGE


def test_model_runs_before_safety_rules():
    order = []

    def model(*args):
        order.append("model")
        return {"prediction": "ROUTINE", "probabilities": confident("ROUTINE")}

    def safety(*args):
        order.append("safety")
        return None

    triage(CLEAN_MESSAGE, CLEAN_INDICATORS, CLEAN_WAITING, predict_fn=model, safety_fn=safety)
    assert order == ["model", "safety"]


# --------------------------------------------------------------------------- #
# 1. Normal valid input / clean confident prediction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label", ["ROUTINE", "MODERATE", "HIGH", "CRITICAL"])
def test_clean_confident_prediction_is_retained(label):
    result = run(label=label)
    assert result["model_prediction"] == label
    assert result["final_recommendation"] == label
    assert result["safety_override_triggered"] is False
    assert result["safety_override_reason"] is None
    assert result["safety_override_floor"] is None
    assert result["input_quality_flags"] == []
    assert result["contradiction_flags"] == []
    assert result["uncertainty"]["needs_clarification"] is False
    assert result["uncertainty"]["reasons"] == []


# --------------------------------------------------------------------------- #
# 2-5. Input quality
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("message", [None, "", "   "])
def test_missing_message(message):
    result = run(message=message)
    assert "message_missing" in result["input_quality_flags"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_very_short_message():
    result = run(message="help")
    assert "message_too_short" in result["input_quality_flags"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_missing_risk_indicators():
    result = run(indicators=None)
    assert "risk_indicators_missing" in result["input_quality_flags"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_none_indicator_is_valid_not_missing():
    result = run(indicators="none")
    assert "risk_indicators_missing" not in result["input_quality_flags"]


def test_missing_waiting_time():
    result = run(waiting=None)
    assert "waiting_time_missing" in result["input_quality_flags"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_zero_waiting_time_is_valid():
    result = run(waiting=0)
    assert "waiting_time_missing" not in result["input_quality_flags"]
    assert result["final_recommendation"] == "ROUTINE"


# --------------------------------------------------------------------------- #
# 6. Low-confidence model prediction
# --------------------------------------------------------------------------- #

def test_low_top_probability():
    result = run(label="ROUTINE", probs=LOW_CONFIDENCE)
    assert result["uncertainty"]["confidence_ok"] is False
    assert "low_top_probability" in result["uncertainty"]["reasons"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_low_probability_margin():
    result = run(label="ROUTINE", probs=LOW_MARGIN)
    assert result["uncertainty"]["confidence_ok"] is False
    assert result["uncertainty"]["confidence_reasons"] == ["low_probability_margin"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


# --------------------------------------------------------------------------- #
# 7. Safety override with low confidence
# --------------------------------------------------------------------------- #

def test_safety_override_with_low_confidence():
    result = run(label="MODERATE", probs=LOW_CONFIDENCE,
                 safety_fn=fake_safety("HIGH", "self_harm"))
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "HIGH"


# --------------------------------------------------------------------------- #
# 8. Contradiction between structured indicators and text
# --------------------------------------------------------------------------- #

def test_structured_indicator_text_mismatch():
    result = run(
        message="Just checking, I am feeling fine and have no symptoms today",
        indicators=["chest_pain"],
    )
    assert result["contradiction_flags"] == ["structured_indicator_text_mismatch"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION
    assert "disagree" in evidence_text(result)


def test_strong_text_no_indicator():
    result = run(
        message="My father is unconscious and not breathing properly",
        indicators=["none"],
    )
    assert result["contradiction_flags"] == ["strong_text_no_indicator"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_insufficient_signal():
    result = run(
        message="I wanted to ask about the opening hours of the clinic",
        indicators=["none"],
    )
    assert result["contradiction_flags"] == ["insufficient_signal"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


# --------------------------------------------------------------------------- #
# 9. Historical-reference uncertainty
# --------------------------------------------------------------------------- #

def test_historical_reference_adds_uncertainty():
    result = run(
        message="I had severe chest pain years ago and it is back now",
        indicators=["chest_pain"],
    )
    assert result["contradiction_flags"] == ["possible_historical_reference"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_historical_reference_does_not_cancel_safety_override():
    result = run(
        message="I had severe chest pain years ago and it is back now",
        indicators=["chest_pain"],
        label="MODERATE",
        safety_fn=fake_safety("CRITICAL", "chest_pain_indicator"),
    )
    assert "possible_historical_reference" in result["contradiction_flags"]
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "CRITICAL"


# --------------------------------------------------------------------------- #
# 10. Multiple safety overrides
# --------------------------------------------------------------------------- #

def test_multiple_safety_overrides_use_highest_floor():
    def multi(model_prediction, risk_indicators, waiting_time):
        return [
            {"triggered": True, "floor": "HIGH", "reason": "self_harm"},
            {"triggered": True, "floor": "CRITICAL", "reason": "plan_indicator"},
            {"triggered": False, "floor": None, "reason": None},
        ]

    result = run(label="ROUTINE", safety_fn=multi)
    assert result["safety_override_floor"] == "CRITICAL"
    assert result["final_recommendation"] == "CRITICAL"
    assert result["safety_override_reasons"] == ["self_harm", "plan_indicator"]
    assert "self_harm" in result["safety_override_reason"]
    assert "plan_indicator" in result["safety_override_reason"]
    text = evidence_text(result)
    assert "self_harm" in text and "plan_indicator" in text


# --------------------------------------------------------------------------- #
# 11. Baseline CRITICAL must never be downgraded
# --------------------------------------------------------------------------- #

def test_critical_model_with_no_override():
    assert run(label="CRITICAL")["final_recommendation"] == "CRITICAL"


def test_critical_model_with_lower_safety_floor_stays_critical():
    result = run(label="CRITICAL", safety_fn=fake_safety("HIGH", "self_harm"))
    assert result["safety_override_floor"] == "HIGH"
    assert result["final_recommendation"] == "CRITICAL"


@pytest.mark.parametrize(
    "indicators, waiting, probs",
    [
        (None, CLEAN_WAITING, None),                 # missing indicators
        (CLEAN_INDICATORS, None, None),              # missing waiting time
        (CLEAN_INDICATORS, CLEAN_WAITING, LOW_CONFIDENCE),  # low confidence
    ],
)
def test_critical_model_with_uncertainty_stays_critical(indicators, waiting, probs):
    result = run(label="CRITICAL", indicators=indicators, waiting=waiting, probs=probs)
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["safety_override_triggered"] is False
    assert result["final_recommendation"] == "CRITICAL"
    assert "never lowered" in evidence_text(result)


# --------------------------------------------------------------------------- #
# 12. Routine input with uncertainty -> NEEDS_CLARIFICATION
# --------------------------------------------------------------------------- #

def test_routine_with_uncertainty_needs_clarification():
    result = run(label="ROUTINE", waiting=None)
    assert result["model_prediction"] == "ROUTINE"
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_needs_clarification_is_not_a_model_class():
    assert NEEDS_CLARIFICATION not in SEVERITY_ORDER
    assert run(label="ROUTINE", waiting=None)["model_prediction"] in SEVERITY_ORDER


# --------------------------------------------------------------------------- #
# Severity precedence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "model_label, floor, expected",
    [
        ("ROUTINE", "HIGH", "HIGH"),
        ("MODERATE", "CRITICAL", "CRITICAL"),   # plan_indicator escalates
        ("HIGH", "HIGH", "HIGH"),               # self_harm, model already HIGH
        ("CRITICAL", "HIGH", "CRITICAL"),       # floors never downgrade
        ("CRITICAL", "MODERATE", "CRITICAL"),
        ("HIGH", "MODERATE", "HIGH"),
    ],
)
def test_severity_precedence(model_label, floor, expected):
    result = run(label=model_label, safety_fn=fake_safety(floor, "some_rule"))
    assert result["final_recommendation"] == expected
    assert result["effective_label"] == expected
    assert SEVERITY_ORDER.index(result["final_recommendation"]) >= \
        SEVERITY_ORDER.index(model_label)


def test_override_escalation_is_explained():
    result = run(label="MODERATE", safety_fn=fake_safety("CRITICAL", "plan_indicator"))
    assert "raised the level from MODERATE to CRITICAL" in evidence_text(result)


# --------------------------------------------------------------------------- #
# Safety override precedence over uncertainty
# --------------------------------------------------------------------------- #

def test_safety_override_beats_uncertainty():
    result = run(
        message=None, indicators=None, waiting=None,
        label="ROUTINE", probs=LOW_CONFIDENCE,
        safety_fn=fake_safety("HIGH", "waiting_time_over_threshold"),
    )
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["final_recommendation"] == "HIGH"
    assert result["final_recommendation"] != NEEDS_CLARIFICATION


def test_no_override_plus_uncertainty_is_needs_clarification():
    result = run(label="MODERATE", waiting=None)
    assert result["safety_override_triggered"] is False
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


# --------------------------------------------------------------------------- #
# Incomplete input where a component cannot run
# --------------------------------------------------------------------------- #

def test_model_failure_on_flagged_input_is_recorded_not_raised():
    def picky_model(message, risk_indicators, waiting_time):
        if not message:
            raise ValueError("empty message")
        return {"prediction": "ROUTINE", "probabilities": confident("ROUTINE")}

    result = triage(None, ["chest_pain"], 10, predict_fn=picky_model, safety_fn=no_safety)
    assert result["model_prediction"] is None
    assert result["model_probabilities"] == {}
    assert result["model_top_probability"] is None
    assert result["component_errors"] == ["baseline_model: ValueError"]
    assert "probabilities_unavailable" in result["uncertainty"]["confidence_reasons"]
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_safety_override_still_applies_when_model_cannot_run():
    def broken_model(message, risk_indicators, waiting_time):
        raise ValueError("no input")

    # With no model prediction available, safety rules receive None first.
    result = triage(None, ["self_harm_mention"], 10, predict_fn=broken_model,
                    safety_fn=fake_safety("HIGH", "self_harm_mention"))
    assert result["model_prediction"] is None
    assert result["final_recommendation"] == "HIGH"


def test_component_failure_on_valid_input_is_not_hidden():
    def broken_model(message, risk_indicators, waiting_time):
        raise ValueError("real bug")

    with pytest.raises(ValueError):
        triage(CLEAN_MESSAGE, CLEAN_INDICATORS, CLEAN_WAITING,
               predict_fn=broken_model, safety_fn=no_safety)


def test_unknown_model_label_is_rejected():
    def bad_model(message, risk_indicators, waiting_time):
        return {"prediction": "URGENT", "probabilities": {}}

    with pytest.raises(ValueError):
        triage(CLEAN_MESSAGE, CLEAN_INDICATORS, CLEAN_WAITING,
               predict_fn=bad_model, safety_fn=no_safety)


def test_override_without_floor_is_rejected():
    def bad_safety(model_prediction, risk_indicators, waiting_time):
        return {"triggered": True, "floor": None, "reason": "x"}

    with pytest.raises(ValueError):
        triage(CLEAN_MESSAGE, CLEAN_INDICATORS, CLEAN_WAITING,
               predict_fn=fake_model("ROUTINE"), safety_fn=bad_safety)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

def test_every_result_has_evidence_and_disclaimer():
    cases = [
        run(),
        run(message=None),
        run(label="CRITICAL", waiting=None),
        run(label="MODERATE", safety_fn=fake_safety("HIGH", "self_harm")),
        run(label="ROUTINE", probs=LOW_CONFIDENCE),
    ]
    for result in cases:
        assert result["evidence"]
        assert all(isinstance(line, str) and line for line in result["evidence"])
        assert "does not diagnose" in evidence_text(result)


def test_evidence_explains_clean_prediction():
    text = evidence_text(run(label="MODERATE"))
    assert "Baseline model suggests MODERATE" in text
    assert "retained" in text


def test_evidence_explains_safety_override():
    text = evidence_text(run(label="ROUTINE", safety_fn=fake_safety("HIGH", "self_harm")))
    assert "Safety rule triggered: self_harm" in text


def test_evidence_explains_input_quality_and_clarification():
    text = evidence_text(run(waiting=None))
    assert "Waiting time was not provided" in text
    assert "more information is needed" in text


def test_evidence_explains_low_confidence():
    text = evidence_text(run(probs=LOW_CONFIDENCE))
    assert "top probability is below 0.55" in text


def test_evidence_includes_waiting_time_and_escalation_reason():
    result = run(waiting=240, label="MODERATE",
                 safety_fn=fake_safety("HIGH", "waiting_time_over_threshold"))
    text = evidence_text(result)
    assert "Reported waiting time: 240 minutes." in text
    assert "waiting_time_over_threshold" in text
    assert result["final_recommendation"] == "HIGH"


def test_evidence_makes_no_diagnostic_claim():
    text = evidence_text(run(label="CRITICAL")).lower()
    assert "you have" not in text
    assert "diagnosed with" not in text


# --------------------------------------------------------------------------- #
# Default adapter -> src.safety_rules.apply_safety_overrides
# --------------------------------------------------------------------------- #
#
# These tests replace the module `src.safety_rules` with a stand-in that has the
# same function name and signature as the real one, then run triage() through
# the DEFAULT adapter. The escalation thresholds in the stand-in are arbitrary;
# the real rules live in safety_rules.py and are tested in test_safety_rules.py.
# What is verified here is that triage.py calls apply_safety_overrides with the
# model prediction first and combines the result with the correct precedence.

LONG_WAIT = 240


def make_safety_module(calls):
    module = types.ModuleType("src.safety_rules")

    def apply_safety_overrides(model_prediction, risk_indicators, waiting_time_minutes):
        calls.append((model_prediction, risk_indicators, waiting_time_minutes))
        indicators = risk_indicators if isinstance(risk_indicators, list) else [risk_indicators]
        hits = []
        if "plan_indicator" in indicators:
            hits.append({"triggered": True, "floor": "CRITICAL", "reason": "plan_indicator"})
        if "self_harm_mention" in indicators:
            hits.append({"triggered": True, "floor": "HIGH", "reason": "self_harm_mention"})
        if (
            model_prediction == "MODERATE"
            and isinstance(waiting_time_minutes, (int, float))
            and waiting_time_minutes >= LONG_WAIT
        ):
            hits.append({"triggered": True, "floor": "HIGH",
                         "reason": "long_waiting_time_moderate"})
        return hits

    module.apply_safety_overrides = apply_safety_overrides
    return module


@contextmanager
def patched_safety_module(calls):
    original = sys.modules.get("src.safety_rules")
    sys.modules["src.safety_rules"] = make_safety_module(calls)
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("src.safety_rules", None)
        else:
            sys.modules["src.safety_rules"] = original


def run_default_safety(label, indicators=CLEAN_INDICATORS, waiting=CLEAN_WAITING,
                       message=CLEAN_MESSAGE, probs=None):
    """Run triage with the real adapter path and the patched safety module."""
    calls = []
    with patched_safety_module(calls):
        result = triage(message, indicators, waiting, predict_fn=fake_model(label, probs))
    return result, calls


def test_apply_safety_overrides_receives_model_prediction_first():
    result, calls = run_default_safety("MODERATE", indicators=["chest_pain"], waiting=33)
    assert calls == [("MODERATE", ["chest_pain"], 33)]
    assert calls[0][0] != CLEAN_MESSAGE
    assert result["model_prediction"] == "MODERATE"


@pytest.mark.parametrize("label", ["ROUTINE", "MODERATE", "HIGH", "CRITICAL"])
def test_each_model_label_is_passed_to_apply_safety_overrides(label):
    _, calls = run_default_safety(label)
    assert calls[0][0] == label


def test_default_adapter_names_the_existing_function():
    assert triage_module._SAFETY_FUNCTION == "apply_safety_overrides"


def test_moderate_with_long_waiting_time_escalates_to_high():
    result, calls = run_default_safety("MODERATE", waiting=LONG_WAIT)
    assert calls[0][0] == "MODERATE"
    assert result["safety_override_triggered"] is True
    assert result["safety_override_floor"] == "HIGH"
    assert result["final_recommendation"] == "HIGH"
    assert "long_waiting_time_moderate" in evidence_text(result)
    assert "raised the level from MODERATE to HIGH" in evidence_text(result)


def test_moderate_with_short_waiting_time_is_not_escalated():
    result, _ = run_default_safety("MODERATE", waiting=15)
    assert result["safety_override_triggered"] is False
    assert result["final_recommendation"] == "MODERATE"


def test_plan_indicator_escalates_to_critical():
    result, _ = run_default_safety("MODERATE", indicators=["plan_indicator"])
    assert result["safety_override_floor"] == "CRITICAL"
    assert result["final_recommendation"] == "CRITICAL"
    assert "plan_indicator" in evidence_text(result)


def test_self_harm_mention_escalates_to_high():
    result, _ = run_default_safety(
        "ROUTINE",
        indicators=["self_harm_mention"],
        message="I have been feeling very low and cannot cope anymore today",
    )
    assert result["safety_override_floor"] == "HIGH"
    assert result["final_recommendation"] == "HIGH"
    assert "self_harm_mention" in evidence_text(result)


def test_self_harm_mention_never_downgrades_critical_model():
    result, calls = run_default_safety("CRITICAL", indicators=["self_harm_mention"])
    assert calls[0][0] == "CRITICAL"
    assert result["safety_override_floor"] == "HIGH"
    assert result["final_recommendation"] == "CRITICAL"


def test_baseline_critical_is_never_downgraded_by_default_adapter():
    for indicators, waiting in [
        (CLEAN_INDICATORS, CLEAN_WAITING),
        (["self_harm_mention"], CLEAN_WAITING),
        (None, None),                       # uncertainty, no override
    ]:
        result, _ = run_default_safety("CRITICAL", indicators=indicators, waiting=waiting)
        assert result["final_recommendation"] == "CRITICAL"


def test_safety_override_wins_when_uncertainty_exists():
    result, _ = run_default_safety(
        "MODERATE",
        indicators=["plan_indicator"],
        waiting=None,                       # missing waiting time -> uncertainty
        probs=LOW_CONFIDENCE,               # low confidence -> uncertainty
    )
    assert result["uncertainty"]["needs_clarification"] is True
    assert "waiting_time_missing" in result["input_quality_flags"]
    assert result["uncertainty"]["confidence_ok"] is False
    assert result["safety_override_triggered"] is True
    assert result["final_recommendation"] == "CRITICAL"


def test_no_override_plus_uncertainty_gives_needs_clarification():
    result, calls = run_default_safety("MODERATE", waiting=None)
    assert calls[0][0] == "MODERATE"
    assert result["safety_override_triggered"] is False
    assert result["uncertainty"]["needs_clarification"] is True
    assert result["final_recommendation"] == NEEDS_CLARIFICATION


def test_no_override_and_no_uncertainty_keeps_model_prediction():
    result, _ = run_default_safety("MODERATE")
    assert result["safety_override_triggered"] is False
    assert result["final_recommendation"] == "MODERATE"


def test_safety_module_is_restored_after_patching():
    before = sys.modules.get("src.safety_rules")
    run_default_safety("ROUTINE")
    assert sys.modules.get("src.safety_rules") is before


# --------------------------------------------------------------------------- #
# Human review queue
# --------------------------------------------------------------------------- #

def test_every_case_is_queued_for_human_review():
    assert run()["human_review"] == {"required": True, "priority": "ROUTINE"}
    assert run(label="ROUTINE", waiting=None)["human_review"]["required"] is True
    escalated = run(label="MODERATE", safety_fn=fake_safety("CRITICAL", "plan_indicator"))
    assert escalated["human_review"]["priority"] == "CRITICAL"


# --------------------------------------------------------------------------- #
# Determinism and isolation
# --------------------------------------------------------------------------- #

def test_deterministic_output():
    kwargs = dict(
        message="I had chest pain years ago",
        indicators=["none"],
        waiting=None,
        label="MODERATE",
        probs=LOW_CONFIDENCE,
        safety_fn=fake_safety("HIGH", "self_harm"),
    )
    assert run(**kwargs) == run(**kwargs)


def test_input_probabilities_are_not_mutated():
    probs = confident("HIGH")
    snapshot = dict(probs)
    run(label="HIGH", probs=probs)
    assert probs == snapshot


def test_triage_module_has_no_direct_ml_imports():
    """Top-level imports must stay light; src.predict is loaded lazily."""
    tree = ast.parse(Path(triage_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = ("predict", "safety_rules", "sklearn", "joblib", "pandas", "numpy")
    for name in imported:
        assert not any(bad in name for bad in forbidden), name
