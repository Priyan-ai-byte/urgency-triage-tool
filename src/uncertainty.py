"""
Uncertainty layer for the Human-Reviewed Urgency Triage Tool.

This module is deterministic and uses only the standard library. It makes no
ML predictions and does not touch the baseline model, the dataset,
`predict.py` or `safety_rules.py`.

It answers one question: "should a human clarify this case before relying on
the automated result?" It never returns an urgency class. NEEDS_CLARIFICATION
is a boolean flag here, not an ML class.

Safety guarantee: the output has no field that can downgrade or cancel a
safety override. Contradiction flags (including possible_historical_reference)
only ADD uncertainty. A later orchestration step must apply safety overrides
independently of this result.

Result shape:
    {
        "input_quality_flags": [],
        "contradiction_flags": [],
        "confidence_ok": True,
        "confidence_reasons": [],
        "needs_clarification": False,
        "reasons": [],
    }
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MIN_TOP_PROBABILITY = 0.55
MIN_MARGIN = 0.10

# Input-quality thresholds
MIN_MESSAGE_TOKENS = 4   # fewer whitespace-separated tokens than this -> message_too_short
MIN_CONTENT_WORDS = 2    # fewer meaningful words than this -> message_low_content

# Input-quality flag names
MESSAGE_MISSING = "message_missing"
MESSAGE_TOO_SHORT = "message_too_short"
MESSAGE_LOW_CONTENT = "message_low_content"
RISK_INDICATORS_MISSING = "risk_indicators_missing"
WAITING_TIME_MISSING = "waiting_time_missing"

# Contradiction flag names
STRUCTURED_INDICATOR_TEXT_MISMATCH = "structured_indicator_text_mismatch"
STRONG_TEXT_NO_INDICATOR = "strong_text_no_indicator"
POSSIBLE_HISTORICAL_REFERENCE = "possible_historical_reference"
INSUFFICIENT_SIGNAL = "insufficient_signal"

# Confidence reason names
LOW_TOP_PROBABILITY = "low_top_probability"
LOW_PROBABILITY_MARGIN = "low_probability_margin"
PROBABILITIES_UNAVAILABLE = "probabilities_unavailable"

# The literal "no risk indicators" value. This is a VALID answer and is
# different from the indicators being absent.
NO_INDICATOR_VALUE = "none"

# Words that carry no triage content (greetings, fillers, function words).
FILLER_WORDS = frozenset(
    {
        "a", "an", "the", "i", "me", "my", "we", "you", "it", "is", "am", "are",
        "was", "to", "of", "and", "or", "in", "on", "at", "for", "with", "this",
        "that", "please", "pls", "thanks", "thank", "thx", "hi", "hello", "hey",
        "ok", "okay", "help", "need", "want", "have", "has", "just", "so",
        "very", "can", "could", "would", "yes", "test", "testing", "asap",
    }
)

# Strong risk phrases, self-contained and independent of safety_rules.py.
_STRONG_TEXT_PATTERNS = [
    r"chest pain",
    r"chest pressure",
    r"heart attack",
    r"can'?t breathe",
    r"cannot breathe",
    r"not breathing",
    r"trouble breathing",
    r"difficulty breathing",
    r"shortness of breath",
    r"unconscious",
    r"unresponsive",
    r"passed out",
    r"seizure",
    r"stroke",
    r"slurred speech",
    r"severe bleeding",
    r"bleeding heavily",
    r"overdose",
    r"suicid\w*",
    r"choking",
]
_STRONG_TEXT_RE = re.compile(r"\b(?:" + "|".join(_STRONG_TEXT_PATTERNS) + r")\b")

# General symptom vocabulary, used only to decide whether the text carries any signal.
_SYMPTOM_TERMS_RE = re.compile(
    r"\b(?:pain\w*|ache\w*|hurt\w*|fever\w*|cough\w*|headache\w*|nause\w*|"
    r"vomit\w*|dizz\w*|bleed\w*|swell\w*|swollen|rash\w*|breath\w*|chest|"
    r"faint\w*|weak\w*|numb\w*|infect\w*|injur\w*|sore|cramp\w*|diarrh\w*|"
    r"burn\w*|wound\w*|fractur\w*|broken|allerg\w*|migraine\w*|palpitation\w*)\b"
)

# Language that suggests nothing is wrong. Used for indicator/text mismatch.
_REASSURANCE_RE = re.compile(
    r"\b(?:feel(?:ing)? fine|i'?m fine|i am fine|i'?m okay|i am okay|"
    r"no symptoms?|nothing(?: is)? wrong|all good|no problem|not a big deal|"
    r"just a question|just checking|feel(?:ing)? better|completely fine)\b"
)

# Conservative historical-reference phrases: only unambiguous past-time wording.
_HISTORICAL_RE = re.compile(
    r"\b(?:"
    r"\d+\s+(?:days?|weeks?|months?|years?)\s+ago|"
    r"(?:a|several|few|many|some)\s+(?:weeks?|months?|years?)\s+ago|"
    r"years?\s+ago|months?\s+ago|"
    r"last\s+(?:year|month)|"
    r"in\s+the\s+past|"
    r"used\s+to|"
    r"history\s+of|"
    r"previously|"
    r"as\s+a\s+(?:child|kid)|"
    r"when\s+i\s+was\s+(?:a\s+)?(?:child|kid|young|little)|"
    r"was\s+diagnosed"
    r")\b"
)

_NEGATION_RE = re.compile(r"\b(?:no|not|without|denies|denied|never)\b|n't\b")
_CLAUSE_SPLIT_RE = re.compile(r"[,.;:!?]|\bbut\b|\band\b")


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #

def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _normalize_message(message: Any) -> Optional[str]:
    """Return the stripped message, or None if missing/blank/not text."""
    if message is None or _is_nan(message) or not isinstance(message, str):
        return None
    stripped = message.strip()
    return stripped if stripped else None


def _normalize_risk_indicators(value: Any) -> Optional[List[str]]:
    """
    Return a list of lowercase indicator tokens, or None if missing.

    Missing: None, NaN, blank string, empty collection, or only blank tokens.
    The literal value "none" is VALID and returned as ["none"].
    """
    if value is None or _is_nan(value):
        return None

    if isinstance(value, str):
        raw_tokens = re.split(r"[,;|]", value)
    elif isinstance(value, (list, tuple, set, frozenset)):
        raw_tokens = [str(v) for v in value if v is not None and not _is_nan(v)]
    else:
        return None

    tokens = [t.strip().lower() for t in raw_tokens if t.strip()]
    return tokens if tokens else None


def _real_indicators(tokens: Optional[List[str]]) -> List[str]:
    """Indicators that actually assert a risk (everything except "none")."""
    if not tokens:
        return []
    return [t for t in tokens if t != NO_INDICATOR_VALUE]


def _is_valid_waiting_time(value: Any) -> bool:
    """A valid waiting time is a finite, non-negative number. 0 is valid."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        if not value.strip():
            return False
        try:
            value = float(value.strip())
        except ValueError:
            return False
    if not isinstance(value, (int, float)):
        return False
    if math.isnan(value) or math.isinf(value):
        return False
    return value >= 0


def _has_non_negated_match(pattern: "re.Pattern[str]", text: str) -> bool:
    """True if the pattern matches somewhere without a nearby negation."""
    for match in pattern.finditer(text):
        window = text[max(0, match.start() - 25):match.start()]
        current_clause = _CLAUSE_SPLIT_RE.split(window)[-1]
        if not _NEGATION_RE.search(current_clause):
            return True
    return False


# --------------------------------------------------------------------------- #
# Input quality
# --------------------------------------------------------------------------- #

def check_input_quality(
    message: Any, risk_indicators: Any, waiting_time: Any
) -> List[str]:
    """Return input-quality flags in a fixed, deterministic order."""
    flags: List[str] = []

    text = _normalize_message(message)
    if text is None:
        flags.append(MESSAGE_MISSING)
    else:
        # 0 tokens is handled above as message_missing; 1-3 tokens is too short.
        if len(text.split()) < MIN_MESSAGE_TOKENS:
            flags.append(MESSAGE_TOO_SHORT)
        words = re.findall(r"[a-z']+", text.lower())
        content_words = [w for w in words if w not in FILLER_WORDS]
        if len(content_words) < MIN_CONTENT_WORDS:
            flags.append(MESSAGE_LOW_CONTENT)

    if _normalize_risk_indicators(risk_indicators) is None:
        flags.append(RISK_INDICATORS_MISSING)

    if not _is_valid_waiting_time(waiting_time):
        flags.append(WAITING_TIME_MISSING)

    return flags


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #

def _extract_probabilities(probabilities: Any) -> Optional[List[float]]:
    """Return probabilities sorted high to low, or None if unusable."""
    if probabilities is None:
        return None
    values = probabilities.values() if isinstance(probabilities, Mapping) else probabilities
    try:
        cleaned = []
        for v in values:
            if isinstance(v, bool):
                return None
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                return None
            cleaned.append(f)
    except (TypeError, ValueError):
        return None
    if len(cleaned) < 2:
        return None
    return sorted(cleaned, reverse=True)


def check_confidence(probabilities: Any) -> Dict[str, Any]:
    """
    Confidence-valid only when
        top_probability >= MIN_TOP_PROBABILITY AND
        top_probability - second_probability >= MIN_MARGIN.

    Values are rounded to 9 decimals before comparing so that float noise
    (e.g. 0.60 - 0.50 = 0.0999999...) cannot flip boundary cases.
    """
    ordered = _extract_probabilities(probabilities)
    if ordered is None:
        return {"confidence_ok": False, "confidence_reasons": [PROBABILITIES_UNAVAILABLE]}

    top = round(ordered[0], 9)
    margin = round(ordered[0] - ordered[1], 9)

    reasons: List[str] = []
    if top < MIN_TOP_PROBABILITY:
        reasons.append(LOW_TOP_PROBABILITY)
    if margin < MIN_MARGIN:
        reasons.append(LOW_PROBABILITY_MARGIN)

    return {"confidence_ok": not reasons, "confidence_reasons": reasons}


# --------------------------------------------------------------------------- #
# Contradictions
# --------------------------------------------------------------------------- #

def check_contradictions(message: Any, risk_indicators: Any) -> List[str]:
    """
    Return contradiction flags. These only ADD uncertainty. They never suppress
    or downgrade a safety override, which is handled elsewhere.

    Text-based checks are skipped when the message is missing, because the
    input-quality layer already flags that case.
    """
    text = _normalize_message(message)
    if text is None:
        return []

    lowered = text.lower()
    tokens = _normalize_risk_indicators(risk_indicators)
    real = _real_indicators(tokens)
    explicit_none = tokens is not None and not real  # valid "none", not missing

    strong_in_text = _has_non_negated_match(_STRONG_TEXT_RE, lowered)
    flags: List[str] = []

    # Structured indicators contradict the text: reassuring language alongside
    # asserted indicators, or "none" listed together with real indicators.
    self_contradictory = (
        tokens is not None and NO_INDICATOR_VALUE in tokens and bool(real)
    )
    reassuring = bool(real) and not strong_in_text and bool(_REASSURANCE_RE.search(lowered))
    if self_contradictory or reassuring:
        flags.append(STRUCTURED_INDICATOR_TEXT_MISMATCH)

    # Strong risk language in the text while indicators explicitly say "none".
    if explicit_none and strong_in_text:
        flags.append(STRONG_TEXT_NO_INDICATOR)

    # Conservative heuristic: only unambiguous past-time phrases.
    if _HISTORICAL_RE.search(lowered):
        flags.append(POSSIBLE_HISTORICAL_REFERENCE)

    # Neither structured indicators nor the text carry any symptom signal.
    has_text_signal = strong_in_text or bool(_SYMPTOM_TERMS_RE.search(lowered))
    if not real and not has_text_signal:
        flags.append(INSUFFICIENT_SIGNAL)

    return flags


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def assess_uncertainty(
    message: Any,
    risk_indicators: Any,
    waiting_time: Any,
    probabilities: Any,
) -> Dict[str, Any]:
    """
    Combine input-quality, confidence and contradiction checks.

    needs_clarification is True if there is at least one input-quality flag,
    OR confidence_ok is False, OR there is at least one contradiction flag.
    """
    input_quality_flags = check_input_quality(message, risk_indicators, waiting_time)
    confidence = check_confidence(probabilities)
    contradiction_flags = check_contradictions(message, risk_indicators)

    confidence_ok = confidence["confidence_ok"]
    confidence_reasons = confidence["confidence_reasons"]

    needs_clarification = bool(
        input_quality_flags or (not confidence_ok) or contradiction_flags
    )

    return {
        "input_quality_flags": input_quality_flags,
        "contradiction_flags": contradiction_flags,
        "confidence_ok": confidence_ok,
        "confidence_reasons": confidence_reasons,
        "needs_clarification": needs_clarification,
        "reasons": input_quality_flags + contradiction_flags + confidence_reasons,
    }
