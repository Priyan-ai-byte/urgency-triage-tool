"""
Triage integration layer for the Human-Reviewed Urgency Triage Tool.

Connects the existing components into one deterministic decision pipeline:

    raw input
      -> input quality check          (src/uncertainty.py)
      -> baseline model               (src/predict.py)
      -> safety override check        (src/safety_rules.py)
      -> contradiction + confidence   (src/uncertainty.py)
      -> final decision logic
      -> evidence / explanation
      -> human review queue entry

This module contains NO ML and NO safety or uncertainty logic of its own. It
only combines results. It is a HUMAN-REVIEW SUPPORT tool: it never diagnoses
and never replaces professional judgment.

Decision rules
--------------
Severity order: ROUTINE < MODERATE < HIGH < CRITICAL.

    effective_label = highest severity of (model prediction, safety floor)

    safety override triggered          -> final = effective_label
    elif effective_label == CRITICAL   -> final = CRITICAL
    elif needs_clarification           -> final = NEEDS_CLARIFICATION
    else                               -> final = effective_label

Safety floors can only escalate; nothing here can lower a model prediction.
A CRITICAL baseline prediction is never replaced by NEEDS_CLARIFICATION.
NEEDS_CLARIFICATION is a decision outcome, not an ML class.

Adapters
--------
The baseline model and the safety rules are called through two small adapters
(`_default_predict_fn`, `_default_safety_fn`) plus normalizers
(`_normalize_model_output`, `_normalize_safety`). If the real signatures or
return shapes of predict.py / safety_rules.py differ from what is assumed
below, adjust only those functions. `triage()` also accepts `predict_fn` and
`safety_fn` so callers (and tests) can inject components.

The two callables are invoked positionally:

    predict_fn(message, risk_indicators, waiting_time_minutes)
    safety_fn(model_prediction, risk_indicators, waiting_time_minutes)

The default safety adapter calls the existing
`src.safety_rules.apply_safety_overrides`, which takes the BASELINE MODEL
PREDICTION (never the raw message) as its first argument. The model therefore
always runs first.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.uncertainty import (
    MIN_MARGIN,
    MIN_TOP_PROBABILITY,
    assess_uncertainty,
    check_input_quality,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SEVERITY_ORDER = ("ROUTINE", "MODERATE", "HIGH", "CRITICAL")
_RANK = {label: i for i, label in enumerate(SEVERITY_ORDER)}

NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

DISCLAIMER = (
    "This tool supports human review. It does not diagnose and does not "
    "replace professional judgment."
)

# Human-readable text for each uncertainty reason.
REASON_TEXT = {
    "message_missing": "No message text was provided.",
    "message_too_short": "The message has fewer than 4 words.",
    "message_low_content": "The message contains little meaningful content.",
    "risk_indicators_missing": (
        "Risk indicators were not provided (this is different from an "
        "explicit 'none')."
    ),
    "waiting_time_missing": (
        "Waiting time was not provided or is not a valid number "
        "(a value of 0 is valid)."
    ),
    "structured_indicator_text_mismatch": (
        "The structured risk indicators and the message text appear to disagree."
    ),
    "strong_text_no_indicator": (
        "The message contains strong risk wording, but the risk indicators say 'none'."
    ),
    "possible_historical_reference": (
        "The message may describe a past event rather than a current one."
    ),
    "insufficient_signal": (
        "Neither the risk indicators nor the message contain a clear symptom signal."
    ),
    "low_top_probability": (
        f"The model's top probability is below {MIN_TOP_PROBABILITY:.2f}."
    ),
    "low_probability_margin": (
        f"The model's top two probabilities are within {MIN_MARGIN:.2f} of each other."
    ),
    "probabilities_unavailable": "Model probabilities were not available.",
}

# Accepted key aliases when normalizing component outputs.
_MODEL_LABEL_KEYS = ("prediction", "label", "predicted_label", "urgency")
_MODEL_PROB_KEYS = ("probabilities", "probs", "probability", "class_probabilities")
_TRIGGER_KEYS = (
    "triggered", "override_triggered", "safety_override_triggered", "override", "applies",
)
_FLOOR_KEYS = (
    "floor", "safety_override_floor", "minimum_label", "min_label", "override_label", "label",
)
_REASON_KEYS = (
    "reason", "reasons", "safety_override_reason", "rule", "rule_name", "rules",
)


# --------------------------------------------------------------------------- #
# Severity helpers
# --------------------------------------------------------------------------- #

def _normalize_label(label: Any, what: str) -> str:
    if not isinstance(label, str) or label.strip().upper() not in _RANK:
        raise ValueError(
            f"Unknown {what} label {label!r}; expected one of {SEVERITY_ORDER}"
        )
    return label.strip().upper()


def max_severity(*labels: Optional[str]) -> Optional[str]:
    """Highest severity among the labels, ignoring None. None if none given."""
    valid = [_normalize_label(lbl, "severity") for lbl in labels if lbl is not None]
    return max(valid, key=_RANK.__getitem__) if valid else None


# --------------------------------------------------------------------------- #
# Adapters: baseline model and safety rules
# --------------------------------------------------------------------------- #

_PREDICT_CANDIDATES = ("predict_urgency", "predict_message", "predict_one", "predict")
_SAFETY_FUNCTION = "apply_safety_overrides"  # existing function in src/safety_rules.py


def _resolve(module_name: str, candidates: Sequence[str]) -> Callable[..., Any]:
    module = importlib.import_module(module_name)
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        f"None of {tuple(candidates)} found in {module_name}. "
        f"Edit the adapter in src/triage.py to call the real function."
    )


def _default_predict_fn(message, risk_indicators, waiting_time_minutes):
    """Adapter to the existing baseline prediction in src/predict.py."""
    fn = _resolve("src.predict", _PREDICT_CANDIDATES)
    return fn(message, risk_indicators, waiting_time_minutes)


def _default_safety_fn(model_prediction, risk_indicators, waiting_time_minutes):
    """
    Adapter to the existing src.safety_rules.apply_safety_overrides.

    The first argument is the baseline model prediction, not the message.
    """
    fn = _resolve("src.safety_rules", (_SAFETY_FUNCTION,))
    return fn(model_prediction, risk_indicators, waiting_time_minutes)


def _first(mapping: Mapping, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _normalize_model_output(result: Any) -> Dict[str, Any]:
    """Return {"label": str, "probabilities": {LABEL: float}} from a model result."""
    if isinstance(result, Mapping):
        label = _first(result, _MODEL_LABEL_KEYS)
        probs = _first(result, _MODEL_PROB_KEYS)
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        label, probs = result
    elif isinstance(result, str):
        label, probs = result, None
    else:
        raise TypeError(f"Unsupported baseline model result: {type(result).__name__}")

    clean: Dict[str, float] = {}
    if isinstance(probs, Mapping):
        for key, value in probs.items():
            clean[str(key).strip().upper()] = float(value)

    return {"label": _normalize_label(label, "model"), "probabilities": clean}


def _normalize_safety(result: Any) -> Dict[str, Any]:
    """
    Return {"triggered": bool, "floor": str|None, "reasons": [str]}.

    Accepts None (no override), one dict, or a list of dicts (multiple rules).
    With several triggered rules the floor is the highest severity.
    """
    if result is None:
        items: List[Any] = []
    elif isinstance(result, Mapping):
        items = [result]
    elif isinstance(result, (list, tuple)):
        items = list(result)
    else:
        raise TypeError(f"Unsupported safety result: {type(result).__name__}")

    floors: List[str] = []
    reasons: List[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise TypeError(f"Unsupported safety rule entry: {type(item).__name__}")
        floor = _first(item, _FLOOR_KEYS)
        triggered = _first(item, _TRIGGER_KEYS)
        if triggered is None:
            triggered = floor is not None
        if not triggered:
            continue
        if floor is None:
            raise ValueError("A safety override was triggered without a floor label")
        floors.append(_normalize_label(floor, "safety floor"))

        reason = _first(item, _REASON_KEYS)
        if isinstance(reason, (list, tuple)):
            new = [str(r) for r in reason]
        elif reason:
            new = [str(reason)]
        else:
            new = ["unspecified safety rule"]
        for r in new:
            if r not in reasons:
                reasons.append(r)

    if not floors:
        return {"triggered": False, "floor": None, "reasons": []}
    return {"triggered": True, "floor": max_severity(*floors), "reasons": reasons}


def _call_component(name, fn, args, tolerate, errors):
    """
    Run a component. If the input is already flagged as incomplete, a failure
    is recorded instead of raised (the case then goes to a human anyway).
    Failures on otherwise valid input are real bugs and are re-raised.
    """
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - deliberate, see docstring
        if not tolerate:
            raise
        errors.append(f"{name}: {type(exc).__name__}")
        return None


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

def _format_waiting(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number) or number < 0:
        return None
    return f"{number:g}"


def _build_evidence(
    *,
    model_label, top, margin, floor, safety_reasons, override_triggered,
    input_quality_flags, contradiction_flags, confidence_reasons,
    waiting_time_minutes, effective_label, final, needs_clarification, errors,
) -> List[str]:
    evidence: List[str] = []

    # Baseline model
    if model_label is None:
        evidence.append("The baseline model could not produce a prediction for this input.")
    else:
        detail = ""
        if top is not None and margin is not None:
            detail = f" (top probability {top:.2f}, margin {margin:.2f})"
        evidence.append(
            f"Baseline model suggests {model_label}{detail}. "
            f"This is a statistical estimate, not a diagnosis."
        )
    for err in errors:
        evidence.append(f"Component issue on incomplete input: {err}.")

    # Waiting time as reported
    waiting = _format_waiting(waiting_time_minutes)
    if waiting is not None:
        evidence.append(f"Reported waiting time: {waiting} minutes.")

    # Safety overrides
    if override_triggered:
        for reason in safety_reasons:
            evidence.append(f"Safety rule triggered: {reason}.")
        if model_label is not None and _RANK[floor] > _RANK[model_label]:
            evidence.append(
                f"The safety rules raised the level from {model_label} to {floor}."
            )
        elif model_label is not None:
            evidence.append(
                f"The model prediction {model_label} is already at or above the "
                f"safety floor {floor}; the level was not lowered."
            )
        else:
            evidence.append(f"The safety rules set a minimum level of {floor}.")

    # Uncertainty
    for flag in input_quality_flags:
        evidence.append(f"Input quality: {REASON_TEXT.get(flag, flag)}")
    for flag in contradiction_flags:
        evidence.append(f"Contradiction check: {REASON_TEXT.get(flag, flag)}")
    for reason in confidence_reasons:
        evidence.append(f"Confidence: {REASON_TEXT.get(reason, reason)}")

    # Decision explanation
    if override_triggered:
        line = (
            f"Final recommendation {final} is the higher of the model prediction "
            f"and the safety floor."
        )
        if needs_clarification:
            line += (
                " Uncertainty was also found; it does not change a safety "
                "override, but a human should review the flagged issues."
            )
        evidence.append(line)
    elif final == NEEDS_CLARIFICATION:
        line = (
            "No safety rule was triggered and uncertainty was found, so more "
            "information is needed before relying on the automated level."
        )
        if effective_label is not None:
            line += f" Highest level suggested so far: {effective_label}."
        evidence.append(line)
    elif needs_clarification:
        evidence.append(
            f"The model prediction {final} is kept: a CRITICAL prediction is "
            f"never lowered because of uncertainty. A human should review the "
            f"flagged issues."
        )
    else:
        evidence.append(
            f"No safety rule was triggered and no uncertainty was found, so the "
            f"baseline model prediction {final} is retained."
        )

    evidence.append(DISCLAIMER)
    return evidence


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def triage(
    message: Any,
    risk_indicators: Any,
    waiting_time_minutes: Any,
    *,
    predict_fn: Optional[Callable[..., Any]] = None,
    safety_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run the full triage pipeline and return a structured, explainable result."""
    predict_fn = predict_fn or _default_predict_fn
    safety_fn = safety_fn or _default_safety_fn
    args = (message, risk_indicators, waiting_time_minutes)

    # 1. Input quality (decides whether component failures may be tolerated)
    input_quality_flags = check_input_quality(*args)
    tolerate = bool(input_quality_flags)
    errors: List[str] = []

    # 2. Baseline model (existing implementation, not duplicated)
    raw_model = _call_component("baseline_model", predict_fn, args, tolerate, errors)
    model_label: Optional[str] = None
    probabilities: Dict[str, float] = {}
    if raw_model is not None:
        normalized = _normalize_model_output(raw_model)
        model_label = normalized["label"]
        probabilities = normalized["probabilities"]

    top: Optional[float] = None
    margin: Optional[float] = None
    if len(probabilities) >= 2:
        ordered = sorted(probabilities.values(), reverse=True)
        top = round(ordered[0], 6)
        margin = round(ordered[0] - ordered[1], 6)

    # 3. Safety override check (existing apply_safety_overrides). It receives
    #    the baseline MODEL PREDICTION first, not the message. If the model
    #    could not run (only possible on already-flagged input), None is passed.
    safety_args = (model_label, risk_indicators, waiting_time_minutes)
    raw_safety = _call_component("safety_rules", safety_fn, safety_args, tolerate, errors)
    safety = _normalize_safety(raw_safety)

    # 4-5. Contradiction + confidence checks (existing implementation)
    uncertainty = assess_uncertainty(
        message, risk_indicators, waiting_time_minutes, probabilities or None
    )
    needs_clarification = uncertainty["needs_clarification"]

    # 6. Final decision logic. Floors only escalate; never downgrade.
    effective_label = max_severity(model_label, safety["floor"])
    if safety["triggered"]:
        final = effective_label
    elif effective_label == "CRITICAL":
        final = effective_label
    elif needs_clarification or effective_label is None:
        final = NEEDS_CLARIFICATION
    else:
        final = effective_label

    # 7. Evidence
    evidence = _build_evidence(
        model_label=model_label,
        top=top,
        margin=margin,
        floor=safety["floor"],
        safety_reasons=safety["reasons"],
        override_triggered=safety["triggered"],
        input_quality_flags=uncertainty["input_quality_flags"],
        contradiction_flags=uncertainty["contradiction_flags"],
        confidence_reasons=uncertainty["confidence_reasons"],
        waiting_time_minutes=waiting_time_minutes,
        effective_label=effective_label,
        final=final,
        needs_clarification=needs_clarification,
        errors=errors,
    )

    # 8. Human review queue entry (every case is reviewed by a person; the
    #    priority is the highest severity suggested so far).
    return {
        "model_prediction": model_label,
        "model_probabilities": dict(probabilities),
        "model_top_probability": top,
        "model_margin": margin,
        "safety_override_triggered": safety["triggered"],
        "safety_override_reason": "; ".join(safety["reasons"]) or None,
        "safety_override_reasons": list(safety["reasons"]),
        "safety_override_floor": safety["floor"],
        "input_quality_flags": list(uncertainty["input_quality_flags"]),
        "contradiction_flags": list(uncertainty["contradiction_flags"]),
        "effective_label": effective_label,
        "final_recommendation": final,
        "uncertainty": uncertainty,
        "human_review": {"required": True, "priority": effective_label},
        "component_errors": errors,
        "evidence": evidence,
    }
