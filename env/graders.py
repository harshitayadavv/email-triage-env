from __future__ import annotations
import os
import re
from typing import List, Dict
from scipy.stats import spearmanr
from openai import OpenAI

from env.data import COMPLAINT_ISSUES


# ── Task 1 grader — Email labeling ──────────────────────────────────────────

def score_labeling(
    agent_labels: Dict[str, str],
    ground_truth: Dict[str, str],
) -> tuple[float, str]:
    """
    Score the agent's label assignments against ground truth.

    Args:
        agent_labels   : {email_id: label} from agent actions
        ground_truth   : {email_id: true_label} from data.py

    Returns:
        (score 0.0-1.0, reason string)
    """
    if not ground_truth:
        return 0.0, "No ground truth available"

    total = len(ground_truth)
    correct = 0
    details = []

    for email_id, true_label in ground_truth.items():
        agent_label = agent_labels.get(email_id)
        if agent_label is None:
            details.append(f"{email_id}: not labeled")
        elif agent_label == true_label:
            correct += 1
            details.append(f"{email_id}: correct ({true_label})")
        else:
            details.append(f"{email_id}: wrong (got {agent_label}, expected {true_label})")

    score = round(correct / total, 4)
    reason = f"{correct}/{total} correct. " + " | ".join(details)
    return score, reason


# ── Task 2 grader — Priority ranking ────────────────────────────────────────

def score_ranking(
    agent_order: List[str],
    ground_truth_order: List[str],
) -> tuple[float, str]:
    """
    Score the agent's ranked list vs ground truth using Spearman correlation.

    Args:
        agent_order        : email ids ordered by agent (most urgent first)
        ground_truth_order : email ids ordered by true priority

    Returns:
        (score 0.0-1.0, reason string)
    """
    if not agent_order or not ground_truth_order:
        return 0.0, "Empty ranking submitted"

    # Only score emails that appear in both lists
    common = [eid for eid in ground_truth_order if eid in agent_order]

    if len(common) < 2:
        return 0.0, "Too few matching email ids to compute correlation"

    # Convert to rank positions (lower index = higher priority = lower rank number)
    gt_ranks = {eid: i for i, eid in enumerate(ground_truth_order)}
    ag_ranks = {eid: i for i, eid in enumerate(agent_order)}

    gt_vector = [gt_ranks[eid] for eid in common]
    ag_vector = [ag_ranks[eid] for eid in common]

    correlation, _ = spearmanr(gt_vector, ag_vector)

    # Clamp to 0.0-1.0 (negative correlation = completely wrong = 0)
    score = round(max(0.0, float(correlation)), 4)

    reason = (
        f"Spearman r = {correlation:.4f} → score = {score}. "
        f"Evaluated {len(common)}/{len(ground_truth_order)} emails. "
        f"Agent order: {agent_order[:5]}... "
        f"Ground truth: {ground_truth_order[:5]}..."
    )
    return score, reason


# ── Task 3 grader — Complaint reply (LLM-as-judge) ──────────────────────────

def score_reply(reply_body: str) -> tuple[float, str]:
    """
    Score the agent's reply using an LLM judge.
    The judge checks whether all 3 complaint issues were addressed.

    Args:
        reply_body : the text of the agent's reply email

    Returns:
        (score 0.0-1.0, reason string)
    """
    if not reply_body or len(reply_body.strip()) < 20:
        return 0.0, "Reply is too short or empty"

    # Build judge prompt
    issues_text = "\n".join(
        f"{i+1}. {issue}" for i, issue in enumerate(COMPLAINT_ISSUES)
    )

    judge_prompt = f"""You are an expert evaluator assessing whether a customer support reply adequately addresses all reported issues.

The customer complaint contained exactly 3 issues:
{issues_text}

The agent's reply is:
---
{reply_body}
---

For each issue, determine if the reply explicitly acknowledges and addresses it.
Respond with ONLY a JSON object in this exact format, nothing else:
{{
  "issue_1_addressed": true or false,
  "issue_2_addressed": true or false,
  "issue_3_addressed": true or false,
  "reasoning": "brief explanation"
}}"""

    try:
        client = OpenAI(
            api_key=os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        )

        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Parse JSON safely
        raw_clean = re.sub(r"```json|```", "", raw).strip()
        import json
        result = json.loads(raw_clean)

        addressed = sum([
            bool(result.get("issue_1_addressed", False)),
            bool(result.get("issue_2_addressed", False)),
            bool(result.get("issue_3_addressed", False)),
        ])

        score = round(addressed / 3, 4)
        reason = (
            f"{addressed}/3 issues addressed. "
            f"Reasoning: {result.get('reasoning', 'none')}. "
            f"Details: billing={result.get('issue_1_addressed')}, "
            f"app={result.get('issue_2_addressed')}, "
            f"callback={result.get('issue_3_addressed')}"
        )
        return score, reason

    except Exception as e:
        # Fallback: keyword matching if LLM call fails
        return _score_reply_fallback(reply_body, str(e))


def _score_reply_fallback(reply_body: str, error: str) -> tuple[float, str]:
    """
    Keyword-based fallback grader if LLM judge is unavailable.
    Less accurate but always works — prevents grader from crashing.
    """
    body_lower = reply_body.lower()

    keywords_per_issue = [
        ["billing", "charged twice", "double charge", "refund", "invoice", "duplicate"],
        ["app", "crash", "mobile", "export", "report", "broken", "bug"],
        ["callback", "call back", "support", "contact", "response", "6 days", "tuesday"],
    ]

    addressed = 0
    details = []
    for i, keywords in enumerate(keywords_per_issue):
        found = any(kw in body_lower for kw in keywords)
        addressed += int(found)
        details.append(f"issue_{i+1}={'yes' if found else 'no'}")

    score = round(addressed / 3, 4)
    reason = (
        f"Fallback grader used (LLM error: {error[:60]}). "
        f"{addressed}/3 issues detected by keywords. "
        + " | ".join(details)
    )
    return score, reason