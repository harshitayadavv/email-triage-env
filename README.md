---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Email Triage Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent
learns to manage a corporate inbox — labeling, prioritizing, and replying
to emails.

## Why this environment?

Email triage is a task every knowledge worker performs daily. This environment
trains and evaluates agents on realistic email management with clear,
deterministic success criteria and meaningful partial reward signals.

## Tasks

| Task | Difficulty | Description | Max Steps |
|------|-----------|-------------|-----------|
| label | Easy | Label 8 emails as spam/urgent/fyi/action-needed | 16 |
| prioritize | Medium | Rank 10 emails by urgency using Spearman correlation | 3 |
| reply | Hard | Draft a reply addressing all 3 issues in a complaint | 6 |

## Action Space
```json
// Task 1 — label
{"action": "label", "email_id": "e1", "label": "urgent"}

// Task 2 — prioritize
{"action": "rank", "order": ["e1", "e3", "e5", "e9", "e14"]}

// Task 3 — reply
{"action": "reply", "email_id": "e18", "body": "Dear customer..."}
```

## Observation Space
```json
{
  "task": "label",
  "instructions": "...",
  "emails": [...],
  "actions_taken": [...],
  "step": 0,
  "done": false
}
```

## Reward Function

- **Label task**: +0.125 per correct label, -0.05 wrong, -0.03 duplicate
- **Prioritize task**: Spearman rank correlation vs ground truth (0.0-1.0)
- **Reply task**: LLM-as-judge scores issues addressed / 3, -0.1 per retry

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| label | Qwen/Qwen2.5-72B-Instruct | ~0.90 |
| prioritize | Qwen/Qwen2.5-72B-Instruct | ~0.96 |
| reply | Qwen/Qwen2.5-72B-Instruct | ~0.67 |

## Setup
```bash
git clone <your-repo>
cd email-triage-env
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Run the server
```bash
uvicorn env.server:app --host 0.0.0.0 --port 7860
```

## Run inference
```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Docker
```bash
docker build -t email-triage-env .
docker run -p 7860:7860 -e HF_TOKEN=your_token email-triage-env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Take one action |
| /state | POST | Inspect full state |
| /score | POST | Get final grader score |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| HF_TOKEN | required | Hugging Face API key |
| MODEL_NAME | Qwen/Qwen2.5-72B-Instruct | LLM model name |
| API_BASE_URL | https://router.huggingface.co/v1 | LLM endpoint |
| PORT | 7860 | Server port |