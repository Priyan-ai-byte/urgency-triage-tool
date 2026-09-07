"""
generate_synthetic_data.py

Stage 1 deliverable for the Human-Reviewed Urgency Triage Tool project.

Generates a fully SYNTHETIC intake dataset for a community mental-health
service, using deterministic, template-based text generation (no external
LLM/API calls). The dataset intentionally includes noisy, ambiguous, and
imperfect records so later project stages (safety rules, uncertainty
handling, evaluation) have real material to work with.

IMPORTANT:
- No real client/patient data is used anywhere in this script.
- All messages are authored from templates + light randomised variation.
- This script only builds the dataset. It does NOT train a model, run any
  UI, or make any triage decision.

Run from the project root with:
    python data/generate_synthetic_data.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

RANDOM_SEED = 42

TOTAL_ROWS = 800

# Target class distribution (must sum to TOTAL_ROWS)
CLASS_COUNTS = {
    "ROUTINE": 320,   # 40%
    "MODERATE": 240,  # 30%
    "HIGH": 160,      # 20%
    "CRITICAL": 80,   # 10%
}

URGENCY_LABELS = ["CRITICAL", "HIGH", "MODERATE", "ROUTINE"]

RISK_INDICATORS = [
    "none",
    "distress",
    "hopelessness",
    "self_harm_mention",
    "immediate_safety_concern",
    "plan_indicator",
    "harm_to_others_indicator",
    "severe_distress",
]

CHANNELS = ["webform", "phone_transcript", "referral_note", "helpline"]

OUTPUT_PATH = Path(__file__).resolve().parent / "synthetic_intake_dataset.csv"


# --------------------------------------------------------------------------
# TEMPLATE BANKS
# --------------------------------------------------------------------------
# Each class has a bank of message templates written in different styles:
# short/medium/long, formal/informal, calm/emotional, vague/specific.
#
# Deliberate design choices (see project spec Section K / T):
# - CRITICAL includes calm, matter-of-fact wording, not only emotional
#   language, so the model cannot rely on "sounds upset" as a shortcut.
# - ROUTINE includes some hyperbolic/non-clinical language ("this is
#   killing me" about admin issues) to create realistic false-positive
#   material.
# - MODERATE includes a historical/resolved risk reference, which is a
#   deliberately ambiguous case for later uncertainty testing.
#
# No message describes a specific method, means, or dosage for self-harm
# or harm to others. Risk is conveyed through general statements
# (e.g. "I have a plan", "I have made arrangements") consistent with how
# real intake/triage systems represent risk without operational detail.

ROUTINE_TEMPLATES = [
    "Hi, I'd like to schedule a follow-up appointment with my counsellor sometime next week.",
    "Just checking in to confirm my appointment time for Thursday afternoon.",
    "I wanted to reschedule my session, is there a slot available next month?",
    "Following up on the referral form I submitted last week, just wanted to check on the status.",
    "Can someone call me back about updating my contact details on file?",
    "need to move my appt pls, mon good?",
    "hey can we talk sometime this week, nothing urgent just wanted to check in",
    "I've been attending sessions for a few months now and things are going okay generally, I just wanted to check if we could move my next appointment slightly later in the day because of a work commitment.",
    "This scheduling system is killing me, can someone just fix my appointment time already?",
    "Wondering if I can switch to a different counsellor going forward, no particular reason.",
    "Quick question - do you offer sessions on weekends?",
    "I missed my appointment yesterday, can we reschedule for later this week?",
    "Just wanted to say thanks for the last session, could we book another one for a similar time next week?",
    "im doing alright, just want to set up a regular check in every couple weeks",
    "Requesting a copy of my intake form for my records, thanks.",
    "Would like to enquire about group session availability for next month.",
    "Can you confirm whether my insurance/coverage details were updated correctly?",
    "I'd like some general information about your services before deciding whether to book.",
    "Following up again, waiting on a callback about rebooking my appointment.",
    "My schedule changed at work, could we move my usual Tuesday slot to Wednesday instead?",
    "Nothing major going on, just wanted to touch base with my counsellor as usual.",
    "Is it possible to do phone sessions instead of in person from now on?",
    "I think I left my jacket in the waiting room last time I came in, is it there?",
    "Could someone send me directions to the office for my appointment next week?",
    "Just a heads up I'll be a few minutes late to my session on Friday.",
    "wanted to check if my referral went through ok",
    "Can I get an update on the waitlist for the parenting support group?",
    "Hoping to switch my appointment to online instead of in person going forward.",
    "This form is honestly killing me, so many fields, can someone just help me finish it over the phone?",
    "All good on my end, just confirming next month's session dates.",
]

MODERATE_TEMPLATES = [
    "I've been feeling really overwhelmed and down for the past couple of weeks and could use someone to talk to soon.",
    "Work stress has been building up and I'm having trouble sleeping most nights.",
    "I feel anxious most of the time lately and it's affecting my daily routine.",
    "Things have been hard since the breakup, I don't feel like myself anymore.",
    "feeling pretty low lately, cud use a chat sometime soon",
    "I used to have thoughts of hurting myself a couple of years ago, it's not like that now but I want to talk to someone about how I've been feeling generally.",
    "just following up again, I've been waiting a while for a callback, been feeling a bit low the past few days",
    "I keep getting into arguments with my family and it's really wearing me down emotionally.",
    "I've noticed I've been withdrawing from friends lately and I'm not sure why.",
    "Lately I feel like I'm just going through the motions, nothing feels enjoyable anymore.",
    "im stressed out all the time and it doesnt seem to be getting better, might need to talk to someone",
    "I've had a couple of panic attacks this month and it's starting to worry me.",
    "Since starting the new job I've felt constantly on edge and exhausted.",
    "I think I might be dealing with some depression, would like to book a proper session to talk it through.",
    "Grief has been hitting me harder than expected since the funeral last month.",
    "I've been crying a lot more than usual and I don't really understand why.",
    "Some days I feel okay and other days I really struggle to get out of bed.",
    "My anxiety has been getting worse and it's starting to affect my ability to concentrate at work.",
    "I had some difficult thoughts a while back but haven't felt that way in months, just want ongoing support.",
    "Things at home have been tense and I feel like I'm constantly walking on eggshells.",
    "not doing great tbh, been feeling kind of hopeless about things lately but i can still function ok",
    "I've been isolating myself more and more over the past month, want to talk about it.",
    "The medication adjustment has left me feeling emotionally flat, would like to discuss it with someone.",
    "I feel like I'm carrying a lot right now and it would help to talk to a professional soon.",
    "My sleep has been all over the place and I think my mood is suffering because of it.",
    "I've felt really discouraged about things lately, nothing specific just a general heaviness.",
    "Started a new medication and my mood has been up and down, could use a check in.",
    "I feel disconnected from everything lately, like I'm watching my life from a distance.",
]

HIGH_TEMPLATES = [
    "I've been having really dark thoughts lately and I don't know what to do anymore.",
    "I feel like everything is falling apart and I can't cope much longer.",
    "I keep thinking about hurting myself, it scares me and I need to talk to someone soon.",
    "I have been thinking about self-harm quite often this week and wanted to flag it before it gets worse.",
    "i cnt keep going like this, feel like giving up on everything, need to talk to someone today",
    "The thoughts of not wanting to be here have been getting stronger over the past few days.",
    "I've started thinking about ways I could hurt myself, I haven't acted on it but it worries me a lot.",
    "Everything feels unbearable right now and I don't know how much longer I can hold on.",
    "I feel like a burden to everyone around me and it's getting harder to push those thoughts away.",
    "I've been having thoughts of hurting my sibling when we argue and I'm scared I might act on it.",
    "My thoughts about ending things have gotten more frequent this week, I need support soon.",
    "I feel numb most of the time but underneath it there's a strong urge to hurt myself.",
    "I haven't been able to shake the thought that everyone would be better off without me.",
    "Things escalated badly at home last night and I said things I'm scared I might actually do.",
    "I've been having intrusive thoughts about self-harm almost every day this week.",
    "I don't feel safe with my own thoughts right now, they keep circling back to hurting myself.",
    "i keep thinking about just not waking up tomorrow, its been getting worse the last few days",
    "The urge to hurt myself has been really strong the last couple of nights.",
    "I feel like I'm right on the edge and I don't trust myself to be alone right now.",
    "I've written some things down that worry me, about not wanting to continue, and I need to talk to someone soon.",
    "I keep having a strong urge to hurt someone close to me and it's frightening me.",
    "My thoughts about self-harm have gotten a lot more specific this week and it's scaring me.",
    "I don't think I can keep pretending I'm okay, the thoughts of hurting myself keep coming back stronger.",
]

CRITICAL_TEMPLATES = [
    "I don't want to be here anymore and I've already made a plan for tonight.",
    "I have decided this is my last day and I have made the necessary arrangements.",
    "I can't see a way out and I intend to end things very soon, I have thought it through.",
    "I am not safe right now and I need someone to talk to me immediately.",
    "I have a plan and today feels like the day I'm going to go through with it.",
    "I've been having thoughts of hurting my partner and I'm scared I'm going to act on it tonight.",
    "I already know how I would do it and I've decided tonight is the night.",
    "im not going to make it through tonight, ive already sorted everything out",
    "I have made arrangements for after I'm gone and I intend to go through with my plan tonight.",
    "I don't see the point in continuing anymore, I already know how I would do it and I've decided.",
    "I need help right now, I have a plan in place and I don't think I can stop myself much longer.",
    "This is it for me, I've made my decision and everything is already arranged.",
    "I'm scared of what I might do to my child right now, I need someone immediately.",
    "I have the means ready and I've decided I'm going through with it tonight.",
    "please someone call me back right away, ive already written letters and im ready to go through with it",
    "I have a specific plan for tonight and I've already put things in place.",
    "I keep thinking about my plan and I've decided today is the day I'm going to act on it.",
    "I am about to hurt someone and I need someone to stop me right now.",
    "I've said goodbye to the people close to me because I've decided this is my last night.",
    "I have everything ready and I don't think I'm going to change my mind this time.",
]

TEMPLATE_BANKS = {
    "ROUTINE": ROUTINE_TEMPLATES,
    "MODERATE": MODERATE_TEMPLATES,
    "HIGH": HIGH_TEMPLATES,
    "CRITICAL": CRITICAL_TEMPLATES,
}

# Very short "low quality input" messages used for a small subset of rows
# (spread mostly across MODERATE/HIGH/CRITICAL to create genuinely
# ambiguous, hard-to-classify short inputs; a couple of ROUTINE ones too).
VERY_SHORT_MESSAGES = {
    "ROUTINE": ["need appt", "call me back", "reschedule pls"],
    "MODERATE": ["not doing great", "need to talk", "feeling low"],
    "HIGH": ["need help now", "cant cope", "scared of my thoughts"],
    "CRITICAL": ["need help now.", "please help me", "not safe right now"],
}


# --------------------------------------------------------------------------
# RISK INDICATOR ASSIGNMENT
# --------------------------------------------------------------------------
# Per-class probability distributions over RISK_INDICATORS.
# Higher urgency classes generally have a higher chance of a stronger
# indicator, but the relationship is deliberately NOT deterministic -
# this is what allows later stages to study false positives/negatives.

RISK_INDICATOR_PROBS = {
    "ROUTINE": {
        "none": 0.70,
        "distress": 0.12,
        "hopelessness": 0.05,
        "self_harm_mention": 0.03,
        "immediate_safety_concern": 0.02,
        "plan_indicator": 0.02,
        "harm_to_others_indicator": 0.02,
        "severe_distress": 0.04,
    },
    "MODERATE": {
        "none": 0.30,
        "distress": 0.28,
        "hopelessness": 0.18,
        "self_harm_mention": 0.10,
        "immediate_safety_concern": 0.03,
        "plan_indicator": 0.03,
        "harm_to_others_indicator": 0.03,
        "severe_distress": 0.05,
    },
    "HIGH": {
        "none": 0.10,
        "distress": 0.15,
        "hopelessness": 0.18,
        "self_harm_mention": 0.25,
        "immediate_safety_concern": 0.07,
        "plan_indicator": 0.10,
        "harm_to_others_indicator": 0.05,
        "severe_distress": 0.10,
    },
    "CRITICAL": {
        "none": 0.05,
        "distress": 0.05,
        "hopelessness": 0.05,
        "self_harm_mention": 0.20,
        "immediate_safety_concern": 0.25,
        "plan_indicator": 0.25,
        "harm_to_others_indicator": 0.10,
        "severe_distress": 0.05,
    },
}

# Probability that a row's indicator is instead drawn completely at random
# from the full indicator pool, ignoring class-conditional probabilities.
# This deliberately manufactures contradictory / ambiguous combinations,
# e.g. a ROUTINE-sounding message paired with "self_harm_mention", or a
# CRITICAL-sounding message paired with "none".
CONTRADICTION_OVERRIDE_PROB = 0.08


# --------------------------------------------------------------------------
# WAITING TIME ASSIGNMENT
# --------------------------------------------------------------------------
# Base ranges (in minutes) per class, only weakly/moderately correlated
# with urgency, plus a chance of an inverted/unusual case.

WAITING_TIME_BASE_RANGES = {
    "CRITICAL": (5, 120),
    "HIGH": (10, 360),
    "MODERATE": (60, 2880),      # up to 2 days
    "ROUTINE": (30, 10080),      # up to 7 days
}

# Probability that waiting time is instead drawn from a "long" or "short"
# override range, independent of class, to create the counter-examples
# requested in the spec (urgent-but-long-waited, mild-but-long-waited, etc).
WAITING_TIME_OVERRIDE_PROB = 0.12
LONG_WAIT_RANGE = (1440, 10080)   # 1 to 7 days
SHORT_WAIT_RANGE = (2, 30)        # 2 to 30 minutes


# --------------------------------------------------------------------------
# CHANNEL ASSIGNMENT
# --------------------------------------------------------------------------
# Channel is weighted per class but not deterministic - every channel is
# possible for every class.

CHANNEL_PROBS = {
    "ROUTINE": {"webform": 0.45, "phone_transcript": 0.20, "referral_note": 0.25, "helpline": 0.10},
    "MODERATE": {"webform": 0.30, "phone_transcript": 0.30, "referral_note": 0.20, "helpline": 0.20},
    "HIGH": {"webform": 0.20, "phone_transcript": 0.35, "referral_note": 0.10, "helpline": 0.35},
    "CRITICAL": {"webform": 0.15, "phone_transcript": 0.35, "referral_note": 0.05, "helpline": 0.45},
}


# --------------------------------------------------------------------------
# TEXT NOISE HELPERS (typos, punctuation/casing variation)
# --------------------------------------------------------------------------

def _swap_adjacent_letters(word: str, rng: random.Random) -> str:
    """Swap two adjacent letters in a word to simulate a typo."""
    if len(word) < 4:
        return word
    i = rng.randint(0, len(word) - 2)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _drop_random_letter(word: str, rng: random.Random) -> str:
    """Drop one letter from a word to simulate a typo."""
    if len(word) < 4:
        return word
    i = rng.randint(0, len(word) - 1)
    return word[:i] + word[i + 1:]


def inject_typo(text: str, rng: random.Random) -> str:
    """Apply a single random typo somewhere in the text, if long enough."""
    words = text.split(" ")
    candidate_indices = [i for i, w in enumerate(words) if len(w) >= 4]
    if not candidate_indices:
        return text
    idx = rng.choice(candidate_indices)
    word = words[idx]
    if rng.random() < 0.5:
        words[idx] = _swap_adjacent_letters(word, rng)
    else:
        words[idx] = _drop_random_letter(word, rng)
    return " ".join(words)


def apply_text_noise(text: str, rng: random.Random) -> str:
    """
    Probabilistically apply small realistic noise to a message:
    - occasional typo
    - occasional dropped final punctuation
    - occasional lowercase-first-letter
    Roughly simulates real-world messy intake text (form fields, quick
    phone transcripts, etc).
    """
    if rng.random() < 0.20:
        text = inject_typo(text, rng)
    if rng.random() < 0.15 and text.endswith((".", "!", "?")):
        text = text[:-1]
    if rng.random() < 0.10 and text:
        text = text[0].lower() + text[1:]
    return text


# --------------------------------------------------------------------------
# ROW GENERATION
# --------------------------------------------------------------------------

def assign_risk_indicator(urgency_label: str, rng: random.Random) -> str:
    if rng.random() < CONTRADICTION_OVERRIDE_PROB:
        return rng.choice(RISK_INDICATORS)
    probs = RISK_INDICATOR_PROBS[urgency_label]
    indicators = list(probs.keys())
    weights = list(probs.values())
    return rng.choices(indicators, weights=weights, k=1)[0]


def assign_waiting_time(urgency_label: str, rng: random.Random) -> int:
    if rng.random() < WAITING_TIME_OVERRIDE_PROB:
        if rng.random() < 0.5:
            low, high = LONG_WAIT_RANGE
        else:
            low, high = SHORT_WAIT_RANGE
    else:
        low, high = WAITING_TIME_BASE_RANGES[urgency_label]
    return rng.randint(low, high)


def assign_channel(urgency_label: str, rng: random.Random) -> str:
    probs = CHANNEL_PROBS[urgency_label]
    channels = list(probs.keys())
    weights = list(probs.values())
    return rng.choices(channels, weights=weights, k=1)[0]


def generate_message(urgency_label: str, rng: random.Random) -> str:
    template = rng.choice(TEMPLATE_BANKS[urgency_label])
    return apply_text_noise(template, rng)


def generate_rows(class_counts: dict, rng: random.Random) -> list:
    """Generate the full list of row dicts (one per synthetic intake)."""
    rows = []
    for urgency_label, count in class_counts.items():
        for _ in range(count):
            message = generate_message(urgency_label, rng)
            risk_indicator = assign_risk_indicator(urgency_label, rng)
            waiting_time = assign_waiting_time(urgency_label, rng)
            channel = assign_channel(urgency_label, rng)

            rows.append(
                {
                    # request_id assigned after shuffling, see build_dataset()
                    "intake_message": message,
                    "risk_indicators": risk_indicator,
                    "waiting_time_minutes": waiting_time,
                    "channel": channel,
                    # Stage 1 simplification: ground-truth professional
                    # outcome and training label are the same value.
                    "professional_triage_outcome": urgency_label,
                    "urgency_label": urgency_label,
                }
            )
    return rows


def inject_missingness(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """
    Introduce a small, controlled proportion of missing values in
    risk_indicators and waiting_time_minutes, to simulate incomplete
    real-world intake forms. Kept small so the dataset stays mostly usable.
    """
    n = len(df)

    missing_risk_idx = rng.sample(range(n), k=max(1, int(0.03 * n)))
    df.loc[missing_risk_idx, "risk_indicators"] = np.nan

    missing_wait_idx = rng.sample(range(n), k=max(1, int(0.03 * n)))
    df.loc[missing_wait_idx, "waiting_time_minutes"] = np.nan

    return df


def inject_very_short_messages(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """
    Replace a small subset of messages with very short, low-information
    text (e.g. "need help"), regardless of their assigned urgency label.
    These rows are intentionally hard/ambiguous to classify from text
    alone and are needed for later uncertainty-handling and edge-case
    testing.
    """
    n = len(df)
    short_idx = rng.sample(range(n), k=max(1, int(0.04 * n)))
    for i in short_idx:
        label = df.loc[i, "urgency_label"]
        df.loc[i, "intake_message"] = rng.choice(VERY_SHORT_MESSAGES[label])
    return df


# --------------------------------------------------------------------------
# DATASET ASSEMBLY
# --------------------------------------------------------------------------

def build_dataset(seed: int = RANDOM_SEED, total_rows: int = TOTAL_ROWS) -> pd.DataFrame:
    py_rng = random.Random(seed)
    np.random.seed(seed)

    assert sum(CLASS_COUNTS.values()) == total_rows, (
        "CLASS_COUNTS must sum to TOTAL_ROWS"
    )

    rows = generate_rows(CLASS_COUNTS, py_rng)

    # Shuffle so classes are not grouped together in the final file.
    py_rng.shuffle(rows)

    df = pd.DataFrame(rows)

    # Assign sequential request_id AFTER shuffling.
    df.insert(0, "request_id", [f"REQ{i:05d}" for i in range(1, len(df) + 1)])

    # Apply controlled imperfections (missingness + very short messages).
    df = inject_missingness(df, py_rng)
    df = inject_very_short_messages(df, py_rng)

    return df


# --------------------------------------------------------------------------
# VALIDATION
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "request_id",
    "intake_message",
    "risk_indicators",
    "waiting_time_minutes",
    "channel",
    "professional_triage_outcome",
    "urgency_label",
]


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Run basic sanity checks on the generated dataset and print a clear
    summary. Returns True if all checks pass, False otherwise.
    """
    checks_passed = True
    print("=" * 60)
    print("Dataset generated successfully.")
    print("=" * 60)
    print(f"\nRows: {len(df)}")

    # Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"\n[FAIL] Missing required columns: {missing_cols}")
        checks_passed = False
    else:
        print("\n[OK] All required columns present.")

    # Class distribution
    print("\nClass distribution (urgency_label):")
    counts = df["urgency_label"].value_counts()
    percentages = df["urgency_label"].value_counts(normalize=True) * 100
    for label in URGENCY_LABELS:
        c = counts.get(label, 0)
        p = percentages.get(label, 0.0)
        print(f"  {label:10s}: {c:4d}  ({p:5.1f}%)")

    # Allowed urgency labels
    bad_labels = set(df["urgency_label"].unique()) - set(URGENCY_LABELS)
    if bad_labels:
        print(f"\n[FAIL] Unexpected urgency labels found: {bad_labels}")
        checks_passed = False
    else:
        print("\n[OK] All urgency_label values are within the allowed set.")

    # Allowed channels
    bad_channels = set(df["channel"].dropna().unique()) - set(CHANNELS)
    if bad_channels:
        print(f"[FAIL] Unexpected channel values found: {bad_channels}")
        checks_passed = False
    else:
        print("[OK] All channel values are within the allowed set.")

    # Duplicate request_id
    n_duplicates = df["request_id"].duplicated().sum()
    print(f"\nDuplicate request_id count: {n_duplicates}")
    if n_duplicates > 0:
        print("[FAIL] Duplicate request_id values found.")
        checks_passed = False
    else:
        print("[OK] request_id values are unique.")

    # Missing values
    print("\nMissing values by column:")
    missing_counts = df.isna().sum()
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col:28s}: {count}")
    if missing_counts.sum() == 0:
        print("  (none)")

    # Waiting time range sanity check
    wt = df["waiting_time_minutes"].dropna()
    if len(wt) > 0:
        print(
            f"\nWaiting time range (minutes): "
            f"min={wt.min():.0f}, max={wt.max():.0f}, mean={wt.mean():.1f}"
        )
        if wt.min() < 0:
            print("[FAIL] Negative waiting time found.")
            checks_passed = False
        else:
            print("[OK] Waiting time values are non-negative.")

    # Row count check
    if len(df) != TOTAL_ROWS:
        print(f"\n[FAIL] Row count {len(df)} does not match expected {TOTAL_ROWS}.")
        checks_passed = False
    else:
        print(f"\n[OK] Row count matches expected total ({TOTAL_ROWS}).")

    print("\n" + "=" * 60)
    print(f"Validation: {'PASS' if checks_passed else 'FAIL'}")
    print("=" * 60)

    return checks_passed


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    df = build_dataset(seed=RANDOM_SEED, total_rows=TOTAL_ROWS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    validate_dataset(df)

    print(f"\nSaved dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
