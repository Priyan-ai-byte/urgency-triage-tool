"""Tests for the uncertainty layer (src/uncertainty.py)."""

import ast
from pathlib import Path

import pytest

from src import uncertainty
from src.uncertainty import (
    MIN_MARGIN,
    MIN_TOP_PROBABILITY,
    assess_uncertainty,
    check_confidence,
    check_contradictions,
    check_input_quality,
)

# A clean, valid baseline input that must produce no uncertainty.
CLEAN_MESSAGE = "Severe chest pain and shortness of breath for the last hour"
CLEAN_INDICATORS = ["chest_pain"]
CLEAN_WAITING = 15
CLEAN_PROBS = {"HIGH": 0.70, "MEDIUM": 0.20, "LOW": 0.10}


def run(message=CLEAN_MESSAGE, indicators=CLEAN_INDICATORS,
        waiting=CLEAN_WAITING, probs=CLEAN_PROBS):
    return assess_uncertainty(message, indicators, waiting, probs)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

def test_thresholds_match_spec():
    assert MIN_TOP_PROBABILITY == 0.55
    assert MIN_MARGIN == 0.10


# --------------------------------------------------------------------------- #
# 1. Missing message
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("message", [None, "", "   ", "\n\t", float("nan")])
def test_missing_message(message):
    result = run(message=message)
    assert "message_missing" in result["input_quality_flags"]
    assert "message_too_short" not in result["input_quality_flags"]
    assert "message_low_content" not in result["input_quality_flags"]
    assert result["needs_clarification"] is True


# --------------------------------------------------------------------------- #
# 2. Too-short message
# --------------------------------------------------------------------------- #

def test_too_short_message():
    result = run(message="help")
    assert "message_too_short" in result["input_quality_flags"]
    assert "message_missing" not in result["input_quality_flags"]
    assert result["needs_clarification"] is True


def test_three_tokens_is_too_short():
    result = run(message="I need help")
    assert "message_too_short" in result["input_quality_flags"]
    assert "message_missing" not in result["input_quality_flags"]
    assert result["needs_clarification"] is True


def test_four_tokens_is_not_too_short():
    flags = check_input_quality("I need urgent help", CLEAN_INDICATORS, CLEAN_WAITING)
    assert "message_too_short" not in flags
    assert "message_missing" not in flags


@pytest.mark.parametrize(
    "message, expected",
    [
        ("help", True),                    # 1 token
        ("chest pain", True),              # 2 tokens, even though 10 characters
        ("I need help", True),             # 3 tokens
        ("I need urgent help", False),     # 4 tokens
        ("  I   need   urgent   help  ", False),  # extra whitespace is ignored
        ("Hi", True),                      # short in characters and tokens
        ("Extraordinarilylongsingleword", True),  # long in characters, 1 token
    ],
)
def test_too_short_is_based_on_whitespace_tokens_not_characters(message, expected):
    flags = check_input_quality(message, CLEAN_INDICATORS, CLEAN_WAITING)
    assert ("message_too_short" in flags) is expected


def test_zero_tokens_is_missing_not_too_short():
    flags = check_input_quality("   ", CLEAN_INDICATORS, CLEAN_WAITING)
    assert flags == ["message_missing"]


# --------------------------------------------------------------------------- #
# 3. Low-content message
# --------------------------------------------------------------------------- #

def test_low_content_message():
    result = run(message="ok thanks hello please thanks")
    assert "message_low_content" in result["input_quality_flags"]
    assert "message_too_short" not in result["input_quality_flags"]
    assert "message_missing" not in result["input_quality_flags"]
    assert result["needs_clarification"] is True


# --------------------------------------------------------------------------- #
# 4. Missing risk indicators (distinct from "none")
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("indicators", [None, "", "   ", [], [""], float("nan")])
def test_missing_risk_indicators(indicators):
    result = run(indicators=indicators)
    assert "risk_indicators_missing" in result["input_quality_flags"]
    assert result["needs_clarification"] is True


@pytest.mark.parametrize("indicators", ["none", "None", " NONE ", ["none"]])
def test_none_indicator_is_not_missing(indicators):
    flags = check_input_quality(CLEAN_MESSAGE, indicators, CLEAN_WAITING)
    assert "risk_indicators_missing" not in flags


# --------------------------------------------------------------------------- #
# 5. Missing waiting time (distinct from 0)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("waiting", [None, "", "  ", "abc", float("nan"), True, -5])
def test_missing_or_invalid_waiting_time(waiting):
    result = run(waiting=waiting)
    assert "waiting_time_missing" in result["input_quality_flags"]
    assert result["needs_clarification"] is True


@pytest.mark.parametrize("waiting", [0, 0.0, "0", 45, "12.5"])
def test_zero_and_numeric_waiting_time_are_valid(waiting):
    flags = check_input_quality(CLEAN_MESSAGE, CLEAN_INDICATORS, waiting)
    assert "waiting_time_missing" not in flags


# --------------------------------------------------------------------------- #
# 6. Valid confidence
# --------------------------------------------------------------------------- #

def test_valid_confidence():
    result = check_confidence({"HIGH": 0.70, "MEDIUM": 0.20, "LOW": 0.10})
    assert result == {"confidence_ok": True, "confidence_reasons": []}


@pytest.mark.parametrize(
    "probs",
    [
        {"HIGH": 0.55, "MEDIUM": 0.45},          # exactly at both thresholds
        {"HIGH": 0.60, "MEDIUM": 0.50},          # float-noise boundary (0.0999...)
        [0.55, 0.45],                            # sequence input
    ],
)
def test_confidence_boundaries_are_valid(probs):
    assert check_confidence(probs)["confidence_ok"] is True


# --------------------------------------------------------------------------- #
# 7. Low top probability
# --------------------------------------------------------------------------- #

def test_low_top_probability():
    result = check_confidence({"HIGH": 0.50, "MEDIUM": 0.30, "LOW": 0.20})
    assert result["confidence_ok"] is False
    assert result["confidence_reasons"] == ["low_top_probability"]


def test_low_top_probability_needs_clarification():
    result = run(probs={"HIGH": 0.50, "MEDIUM": 0.30, "LOW": 0.20})
    assert result["confidence_ok"] is False
    assert result["needs_clarification"] is True
    assert result["input_quality_flags"] == []
    assert result["contradiction_flags"] == []


# --------------------------------------------------------------------------- #
# 8. Low probability margin
# --------------------------------------------------------------------------- #

def test_low_probability_margin():
    result = check_confidence({"HIGH": 0.60, "MEDIUM": 0.52, "LOW": 0.48})
    assert result["confidence_ok"] is False
    assert result["confidence_reasons"] == ["low_probability_margin"]


def test_both_confidence_conditions_fail():
    result = check_confidence({"HIGH": 0.40, "MEDIUM": 0.35, "LOW": 0.25})
    assert result["confidence_ok"] is False
    assert result["confidence_reasons"] == [
        "low_top_probability",
        "low_probability_margin",
    ]


@pytest.mark.parametrize("probs", [None, {}, {"HIGH": 1.0}, [0.9], {"A": "x", "B": 0.5}])
def test_unusable_probabilities_are_not_confident(probs):
    result = check_confidence(probs)
    assert result["confidence_ok"] is False
    assert result["confidence_reasons"] == ["probabilities_unavailable"]


# --------------------------------------------------------------------------- #
# 9. Contradiction detection
# --------------------------------------------------------------------------- #

def test_structured_indicator_text_mismatch():
    message = "Just checking, I am feeling fine and have no symptoms today"
    flags = check_contradictions(message, ["chest_pain"])
    assert flags == ["structured_indicator_text_mismatch"]


def test_mismatch_when_none_listed_with_real_indicator():
    flags = check_contradictions(
        "I have a bad cough and a sore throat", ["none", "chest_pain"]
    )
    assert "structured_indicator_text_mismatch" in flags


def test_strong_text_no_indicator():
    message = "My father is unconscious and not breathing properly"
    flags = check_contradictions(message, ["none"])
    assert flags == ["strong_text_no_indicator"]


def test_strong_text_with_missing_indicators_is_not_this_flag():
    message = "My father is unconscious and not breathing properly"
    flags = check_contradictions(message, None)
    assert "strong_text_no_indicator" not in flags


def test_negated_strong_text_is_not_a_contradiction():
    message = "I do not have chest pain, just a sore throat"
    assert check_contradictions(message, ["none"]) == []


def test_contradiction_flags_trigger_clarification():
    result = run(
        message="My father is unconscious and not breathing properly",
        indicators=["none"],
    )
    assert result["contradiction_flags"] == ["strong_text_no_indicator"]
    assert result["needs_clarification"] is True


def test_contradictions_skipped_for_missing_message():
    assert check_contradictions(None, ["chest_pain"]) == []


# --------------------------------------------------------------------------- #
# 10. Historical-reference uncertainty
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "message",
    [
        "I had severe chest pain years ago and it is back now",
        "I had a stroke 3 years ago and now have a headache",
        "There is a history of seizure in my family and I feel dizzy",
    ],
)
def test_possible_historical_reference(message):
    flags = check_contradictions(message, ["chest_pain"])
    assert flags == ["possible_historical_reference"]


def test_historical_reference_only_adds_uncertainty():
    result = run(message="I had severe chest pain years ago and it is back now")
    assert "possible_historical_reference" in result["contradiction_flags"]
    assert result["needs_clarification"] is True
    # Output carries no field that could cancel or downgrade a safety override.
    assert set(result) == {
        "input_quality_flags",
        "contradiction_flags",
        "confidence_ok",
        "confidence_reasons",
        "needs_clarification",
        "reasons",
    }


def test_current_event_is_not_historical():
    flags = check_contradictions(CLEAN_MESSAGE, CLEAN_INDICATORS)
    assert "possible_historical_reference" not in flags


# --------------------------------------------------------------------------- #
# 11. Insufficient signal
# --------------------------------------------------------------------------- #

def test_insufficient_signal():
    message = "I wanted to ask about the opening hours of the clinic"
    flags = check_contradictions(message, ["none"])
    assert flags == ["insufficient_signal"]


def test_insufficient_signal_triggers_clarification():
    result = run(
        message="I wanted to ask about the opening hours of the clinic",
        indicators=["none"],
    )
    assert "insufficient_signal" in result["contradiction_flags"]
    assert result["needs_clarification"] is True


def test_real_indicator_provides_signal():
    flags = check_contradictions(
        "I wanted to ask about the opening hours of the clinic", ["chest_pain"]
    )
    assert "insufficient_signal" not in flags


def test_symptom_text_provides_signal():
    flags = check_contradictions("I have a mild headache and a runny nose", ["none"])
    assert "insufficient_signal" not in flags


# --------------------------------------------------------------------------- #
# 12. Combined uncertainty conditions
# --------------------------------------------------------------------------- #

def test_everything_missing_and_low_confidence():
    result = assess_uncertainty(
        message=None,
        risk_indicators=None,
        waiting_time=None,
        probabilities={"HIGH": 0.40, "MEDIUM": 0.35, "LOW": 0.25},
    )
    assert result["input_quality_flags"] == [
        "message_missing",
        "risk_indicators_missing",
        "waiting_time_missing",
    ]
    assert result["confidence_ok"] is False
    assert result["confidence_reasons"] == [
        "low_top_probability",
        "low_probability_margin",
    ]
    assert result["needs_clarification"] is True
    assert result["reasons"] == (
        result["input_quality_flags"]
        + result["contradiction_flags"]
        + result["confidence_reasons"]
    )


def test_quality_contradiction_and_confidence_combined():
    result = assess_uncertainty(
        message="I had chest pain years ago",
        risk_indicators=["none"],
        waiting_time=None,
        probabilities={"HIGH": 0.50, "MEDIUM": 0.45, "LOW": 0.05},
    )
    assert result["input_quality_flags"] == ["waiting_time_missing"]
    assert result["contradiction_flags"] == [
        "strong_text_no_indicator",
        "possible_historical_reference",
    ]
    assert result["confidence_reasons"] == [
        "low_top_probability",
        "low_probability_margin",
    ]
    assert result["confidence_ok"] is False
    assert result["needs_clarification"] is True
    assert len(result["reasons"]) == 5


def test_each_trigger_alone_is_sufficient():
    assert run(waiting=None)["needs_clarification"] is True
    assert run(probs={"A": 0.5, "B": 0.3, "C": 0.2})["needs_clarification"] is True
    assert run(
        message="I wanted to ask about the opening hours of the clinic",
        indicators=["none"],
    )["needs_clarification"] is True


# --------------------------------------------------------------------------- #
# 13. No uncertainty for clean valid input
# --------------------------------------------------------------------------- #

def test_clean_valid_input_has_no_uncertainty():
    assert run() == {
        "input_quality_flags": [],
        "contradiction_flags": [],
        "confidence_ok": True,
        "confidence_reasons": [],
        "needs_clarification": False,
        "reasons": [],
    }


def test_clean_input_with_zero_waiting_time_and_none_indicator():
    result = assess_uncertainty(
        message="I have a mild headache and nausea since this morning",
        risk_indicators="none",
        waiting_time=0,
        probabilities={"HIGH": 0.10, "MEDIUM": 0.25, "LOW": 0.65},
    )
    assert result["needs_clarification"] is False
    assert result["reasons"] == []


# --------------------------------------------------------------------------- #
# Determinism and isolation
# --------------------------------------------------------------------------- #

def test_deterministic():
    kwargs = dict(
        message="I had chest pain years ago",
        indicators=["none"],
        waiting=None,
        probs={"HIGH": 0.50, "MEDIUM": 0.45, "LOW": 0.05},
    )
    assert run(**kwargs) == run(**kwargs)


def test_module_does_not_import_forbidden_modules():
    """uncertainty.py must not depend on predict, safety_rules, or ML libraries."""
    tree = ast.parse(Path(uncertainty.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = ("predict", "safety_rules", "sklearn", "joblib", "pandas", "numpy")
    for name in imported:
        assert not any(bad in name for bad in forbidden), name
