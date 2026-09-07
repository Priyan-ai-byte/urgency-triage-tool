"""
test_prediction.py

Stage 2 sanity check: loads the saved baseline pipeline
(models/baseline_model.joblib) via src/predict.py and runs it on a
handful of hand-written example intake records spanning all four
urgency levels, to confirm the saved pipeline can process brand-new raw
input end-to-end (i.e. the full preprocessing + model round-trip works
outside of training).

This is a smoke test, NOT a formal evaluation - see
evaluation/baseline_results.md for the actual measured metrics.
Predictions here are not expected or required to be perfect.

Run from the project root:
    python training/test_prediction.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_urgency  # noqa: E402

EXAMPLES = [
    {
        "description": "1. Clearly ROUTINE request",
        "intake_message": "Hi, I'd like to reschedule my appointment to next Tuesday if possible.",
        "risk_indicators": "none",
        "waiting_time_minutes": 120,
    },
    {
        "description": "2. Clearly MODERATE distress request",
        "intake_message": (
            "I've been feeling really anxious and overwhelmed for the past few weeks "
            "and would like to talk to someone soon."
        ),
        "risk_indicators": "distress",
        "waiting_time_minutes": 300,
    },
    {
        "description": "3. Clearly HIGH urgency request",
        "intake_message": (
            "I keep having thoughts of hurting myself and it's getting harder to push "
            "them away, I need to talk to someone soon."
        ),
        "risk_indicators": "self_harm_mention",
        "waiting_time_minutes": 45,
    },
    {
        "description": "4. Clearly CRITICAL request",
        "intake_message": (
            "I have a plan and I've already decided tonight is the night, I don't "
            "think I can stop myself."
        ),
        "risk_indicators": "plan_indicator",
        "waiting_time_minutes": 10,
    },
]


def main():
    print("Running baseline model smoke test on hand-written examples.\n")
    print("Note: this confirms the saved pipeline works end-to-end on new raw")
    print("input. It is not a formal evaluation, and perfect predictions are")
    print("not expected or required at this stage.\n")

    for example in EXAMPLES:
        result = predict_urgency(
            intake_message=example["intake_message"],
            risk_indicators=example["risk_indicators"],
            waiting_time_minutes=example["waiting_time_minutes"],
        )

        print("-" * 60)
        print(example["description"])
        print(f"Message: {example['intake_message']}")
        print(f"Risk indicator (input): {example['risk_indicators']}")
        print(f"Waiting time (input):   {example['waiting_time_minutes']} minutes")
        print(f"\nPredicted urgency: {result['predicted_label']}")
        print("Probabilities:")
        for label, prob in result["class_probabilities"].items():
            print(f"  {label:10s}: {prob:.3f}")
        print(f"Top probability: {result['top_probability']:.3f}")
        print()


if __name__ == "__main__":
    main()
