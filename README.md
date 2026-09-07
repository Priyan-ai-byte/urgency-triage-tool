# Human-Reviewed Urgency Triage Tool for Community Mental-Health Service Requests

## Problem

Community mental-health services receive counselling/helpline intake requests through
multiple channels, and these are often processed in submission order or by appointment
slot availability. This means a request describing an active safety concern can end up
waiting behind routine appointment requests simply because it arrived later. This project
is a prototype decision-support tool that helps a human reviewer identify which intake
requests may need attention soonest, using a combination of a machine learning model and
transparent risk indicators — while keeping a qualified professional in control of every
final decision.

## Current Project Stage

This repository currently implements **Stage 1 only**: project structure and synthetic
dataset generation. The following are **not yet implemented** and will be added in later
stages: the ML model, the safety override layer, the uncertainty/clarification layer, the
explainability layer, the Streamlit human-review UI, storage, and evaluation.

## Synthetic Data Only

Every record in this dataset is artificially generated from message templates and
randomised variation. **No real client, patient, or service-user data is used or
referenced anywhere in this project.** The dataset generator (`data/generate_synthetic_data.py`)
does not call any external API or LLM — all text is produced locally and deterministically
from a fixed random seed, so running it again reproduces the exact same dataset.

## How to Generate the Dataset

From the project root:

```bash
python data/generate_synthetic_data.py
```

This creates (or overwrites) `data/synthetic_intake_dataset.csv` and prints a validation
summary (row count, class distribution, missing values, duplicate ID check, etc.) to the
console.

## Dataset Fields

| Field | Description |
|---|---|
| `request_id` | Unique synthetic identifier for the intake request |
| `intake_message` | Free-text synthetic intake message |
| `risk_indicators` | Structured synthetic risk tag (e.g. `distress`, `self_harm_mention`, `none`); may be missing |
| `waiting_time_minutes` | Synthetic time (in minutes) already waited; may be missing |
| `channel` | Synthetic intake channel: `webform`, `phone_transcript`, `referral_note`, or `helpline` |
| `professional_triage_outcome` | Synthetic ground-truth outcome, as if assigned by a professional reviewer |
| `urgency_label` | Target class for the baseline model (see below) |

## Urgency Labels

The model (to be built in a later stage) predicts one of four professional urgency
classes:

1. **CRITICAL**
2. **HIGH**
3. **MODERATE**
4. **ROUTINE**

A separate uncertainty/safety layer (added in a later stage) can flag a prediction as
needing clarification or human escalation — this is **not** a model training class, it is
a safeguard layered on top of the model's output.

## Important Safety Statement

**This is not, and will not become, an autonomous clinical decision-making system.** At no
stage does this tool make a final clinical or triage decision on its own. Its only purpose
is to produce a recommendation, with supporting evidence, for review by an authorised
human professional, who retains full authority to confirm, downgrade, escalate, or request
more information before any action is taken.
