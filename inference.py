from __future__ import annotations
import json
import os
import re
import sys
import requests

from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK    = "email-triage-env"
MAX_STEPS    = 20
TEMPERATURE  = 0.2
MAX_TOKENS   = 512

TASKS = ["label", "prioritize", "reply"]

# ── OpenAI client ─────────────────────────────────────────────────────────────

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── System prompts per task ───────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "label": """You are an email triage assistant. Your job is to label emails.

For each step, you must label ONE email by outputting a single JSON object.
Available labels: spam, urgent, fyi, action-needed

Rules:
- spam: unsolicited, promotional, scam emails
- urgent: requires immediate attention, production issues, security alerts
- fyi: informational only, no action needed
- action-needed: requires a response or task but not immediately critical

Respond with ONLY a JSON object, no explanation, no markdown:
{"action": "label", "email_id": "e1", "label": "urgent"}""",

    "prioritize": """You are an email triage assistant. Your job is to rank emails by priority.

You will see a list of emails. Output a single JSON object ranking ALL email ids
from most urgent (first) to least urgent (last).

Consider: production outages > security issues > client complaints > deadlines > admin > fyi > spam

Respond with ONLY a JSON object, no explanation, no markdown:
{"action": "rank", "order": ["e1", "e5", "e3", "e9", "e14"]}""",

    "reply": """You are an email triage assistant. Your job is to reply to a customer complaint.

Read the complaint carefully. It contains exactly 3 issues.
Write a professional reply that explicitly addresses ALL 3 issues.
Be specific — mention each problem and what action is being taken.

Respond with ONLY a JSON object, no explanation, no markdown:
{"action": "reply", "email_id": "e18", "body": "Dear customer, ..."}""",
}

# ── Env API helpers ───────────────────────────────────────────────────────────

def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()

# ── Action parser ─────────────────────────────────────────────────────────────

def parse_action(text: str) -> dict | None:
    """
    Extract JSON action from LLM response.
    Handles raw JSON, markdown code blocks, and extra text.
    """
    if not text:
        return None

    # Strip markdown fences
    text = re.sub(r"```json|```", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object from text
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None

# ── Build prompt from observation ─────────────────────────────────────────────

def build_user_prompt(obs: dict, task: str) -> str:
    """Convert observation dict into a clear prompt for the LLM."""
    emails = obs.get("emails", [])
    actions_taken = obs.get("actions_taken", [])
    step = obs.get("step", 0)

    lines = []
    lines.append(f"Step {step}. Task: {task}")
    lines.append("")
    lines.append("INBOX:")

    for e in emails:
        lines.append(f"  ID: {e['id']}")
        lines.append(f"  From: {e['sender']}")
        lines.append(f"  Subject: {e['subject']}")
        lines.append(f"  Body: {e['body'][:300]}")
        lines.append("")

    if actions_taken:
        lines.append("ACTIONS YOU HAVE ALREADY TAKEN:")
        for a in actions_taken[-5:]:   # show last 5 only
            lines.append(f"  Step {a['step']}: {a['action']}")
        lines.append("")
        lines.append("Do NOT repeat actions on emails you have already labeled.")

    return "\n".join(lines)

# ── Single task runner ────────────────────────────────────────────────────────

def run_task(task: str) -> None:
    """Run one full episode for a task. Emits [START] [STEP] [END] to stdout."""

    rewards: list[float] = []
    last_error: str | None = None
    success = False
    step_num = 0

    # [START]
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        # Reset environment
        reset_resp = env_reset(task)
        obs = reset_resp["observation"]

        system_prompt = SYSTEM_PROMPTS[task]

        for step_num in range(1, MAX_STEPS + 1):
            # Build prompt
            user_prompt = build_user_prompt(obs, task)

            # Call LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                llm_output = response.choices[0].message.content or ""
            except Exception as e:
                llm_output = ""
                last_error = f"LLM error: {str(e)[:80]}"

            # Parse action
            action = parse_action(llm_output)

            if action is None:
                last_error = f"Could not parse action from: {llm_output[:60]}"
                # Emit step with 0 reward and error
                print(
                    f"[STEP] step={step_num} action=null "
                    f"reward=0.00 done=false error={last_error}",
                    flush=True,
                )
                rewards.append(0.0)
                continue

            # Call env step
            try:
                step_resp = env_step(action)
            except Exception as e:
                last_error = f"Env error: {str(e)[:80]}"
                print(
                    f"[STEP] step={step_num} action={json.dumps(action)} "
                    f"reward=0.00 done=false error={last_error}",
                    flush=True,
                )
                rewards.append(0.0)
                continue

            reward = step_resp.get("reward", 0.0)
            done   = step_resp.get("done", False)
            info   = step_resp.get("info", {})
            obs    = step_resp.get("observation", obs)

            step_error = info.get("error") or None
            last_error = step_error

            # [STEP]
            print(
                f"[STEP] step={step_num} "
                f"action={json.dumps(action)} "
                f"reward={reward:.2f} "
                f"done={'true' if done else 'false'} "
                f"error={step_error if step_error else 'null'}",
                flush=True,
            )
            rewards.append(reward)

            if done:
                final_score = info.get("final_score", 0.0)
                success = final_score >= 0.5
                break

    except Exception as e:
        last_error = str(e)

    # [END]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step_num} "
        f"rewards={rewards_str}",
        flush=True,
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Allow running a single task via env var, otherwise run all 3
    single_task = os.getenv("TASK")
    tasks_to_run = [single_task] if single_task else TASKS

    for task in tasks_to_run:
        run_task(task)
        print(flush=True)   # blank line between tasks

if __name__ == "__main__":
    main()