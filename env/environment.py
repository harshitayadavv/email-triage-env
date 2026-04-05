from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from env.models import (
    Email,
    EmailObservation,
    EmailReward,
    EnvState,
    LabelAction,
    RankAction,
    ReplyAction,
)
from env.data import (
    get_task_emails,
    get_ground_truth_labels,
    get_ground_truth_ranking,
)
from env.graders import score_labeling, score_ranking, score_reply


# ── Task configuration ───────────────────────────────────────────────────────

TASK_CONFIG = {
    "label": {
        "max_steps": 16,
        "instructions": (
            "You are an email assistant. You must label each email in the inbox. "
            "For each email, call the label action with the email_id and one of these labels: "
            "spam, urgent, fyi, action-needed. "
            "Label every email exactly once. "
            "Respond with ONLY valid JSON. Example: "
            '{"action": "label", "email_id": "e1", "label": "urgent"}'
        ),
    },
    "prioritize": {
        "max_steps": 3,
        "instructions": (
            "You are an email assistant. You must rank all emails in the inbox "
            "from most urgent to least urgent. "
            "Submit a single rank action containing all email ids in priority order. "
            "Respond with ONLY valid JSON. Example: "
            '{"action": "rank", "order": ["e1", "e5", "e9", "e14"]}'
        ),
    },
    "reply": {
        "max_steps": 6,
        "instructions": (
            "You are an email assistant. Read the customer complaint carefully. "
            "It contains exactly 3 issues. Draft a professional reply that "
            "explicitly acknowledges and addresses all 3 issues. "
            "Respond with ONLY valid JSON. Example: "
            '{"action": "reply", "email_id": "e18", "body": "Dear customer, ..."}'
        ),
    },
}

# Reward constants
LABEL_CORRECT   =  0.125   # 8 emails × 0.125 = 1.0 max
LABEL_WRONG     = -0.05
LABEL_DUPLICATE = -0.03
REPLY_RETRY_PENALTY = 0.10


class EmailTriageEnv:
    """
    Core environment class.
    Manages state, validates actions, computes rewards, runs graders.
    """

    def __init__(self) -> None:
        self._task: Optional[str] = None
        self._inbox: List[Email] = []
        self._actions_taken: List[dict] = []
        self._step_count: int = 0
        self._done: bool = False
        self._total_reward: float = 0.0
        self._reply_attempts: int = 0

        # task-specific tracking
        self._agent_labels: Dict[str, str] = {}   # {email_id: label} for task 1
        self._ground_truth_labels: Dict[str, str] = {}
        self._ground_truth_ranking: List[str] = []

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self, task: str) -> EmailObservation:
        """Reset environment to clean initial state for the given task."""
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task '{task}'. Choose: {list(TASK_CONFIG)}")

        self._task = task
        self._inbox = get_task_emails(task)
        self._actions_taken = []
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._reply_attempts = 0
        self._agent_labels = {}
        self._ground_truth_labels = get_ground_truth_labels(task) if task == "label" else {}
        self._ground_truth_ranking = get_ground_truth_ranking(task) if task == "prioritize" else []

        return self._make_observation()

    def step(
        self, action_data: dict
    ) -> Tuple[EmailObservation, float, bool, Dict[str, Any]]:
        """
        Process one agent action.

        Returns:
            observation : what agent sees next
            reward      : float reward for this step
            done        : whether episode is finished
            info        : extra info dict (error, grader_score, etc.)
        """
        if self._done:
            return self._make_observation(), 0.0, True, {"error": "Episode already done. Call reset()."}

        if self._task is None:
            return self._make_observation(), 0.0, True, {"error": "Call reset() before step()."}

        self._step_count += 1
        info: Dict[str, Any] = {"error": None}
        reward = EmailReward(value=0.0, reason="")

        # ── Parse and route action ───────────────────────────────────────────
        action_type = action_data.get("action")

        if action_type == "label" and self._task == "label":
            reward, info = self._handle_label(action_data)

        elif action_type == "rank" and self._task == "prioritize":
            reward, info = self._handle_rank(action_data)

        elif action_type == "reply" and self._task == "reply":
            reward, info = self._handle_reply(action_data)

        else:
            info["error"] = (
                f"Invalid action '{action_type}' for task '{self._task}'. "
                f"Expected: {self._expected_action()}"
            )
            reward = EmailReward(value=0.0, reason="invalid action type")

        # ── Record action ────────────────────────────────────────────────────
        self._actions_taken.append({
            "step": self._step_count,
            "action": action_data,
            "reward": reward.value,
            "reason": reward.reason,
        })
        self._total_reward += reward.value

        # ── Check step limit ─────────────────────────────────────────────────
        max_steps = TASK_CONFIG[self._task]["max_steps"]
        if self._step_count >= max_steps and not self._done:
            self._done = True
            info["timeout"] = f"Step limit {max_steps} reached"

        return self._make_observation(), round(reward.value, 4), self._done, info

    def state(self) -> EnvState:
        """Return full internal state snapshot."""
        return EnvState(
            task=self._task or "",
            step=self._step_count,
            done=self._done,
            inbox=copy.deepcopy(self._inbox),
            actions_taken=copy.deepcopy(self._actions_taken),
            total_reward=round(self._total_reward, 4),
        )

    def final_score(self) -> Tuple[float, str]:
        """
        Run the appropriate grader and return final episode score.
        Called after done=True.
        """
        if self._task == "label":
            return score_labeling(self._agent_labels, self._ground_truth_labels)

        elif self._task == "prioritize":
            if not self._actions_taken:
                return 0.0, "No ranking submitted"
            last_rank_action = None
            for entry in reversed(self._actions_taken):
                if entry["action"].get("action") == "rank":
                    last_rank_action = entry["action"]
                    break
            if last_rank_action is None:
                return 0.0, "No rank action found"
            return score_ranking(
                last_rank_action["order"],
                self._ground_truth_ranking,
            )

        elif self._task == "reply":
            if not self._actions_taken:
                return 0.0, "No reply submitted"
            last_reply = None
            for entry in reversed(self._actions_taken):
                if entry["action"].get("action") == "reply":
                    last_reply = entry["action"]
                    break
            if last_reply is None:
                return 0.0, "No reply action found"
            return score_reply(last_reply["body"])

        return 0.0, "Unknown task"

    # ── Private handlers ─────────────────────────────────────────────────────

    def _handle_label(
        self, action_data: dict
    ) -> Tuple[EmailReward, Dict[str, Any]]:
        info: Dict[str, Any] = {"error": None}

        try:
            action = LabelAction(**action_data)
        except Exception as e:
            info["error"] = f"Invalid label action format: {e}"
            return EmailReward(value=0.0, reason="parse error"), info

        email_id = action.email_id
        label = action.label

        # Check email exists in inbox
        inbox_ids = {e.id for e in self._inbox}
        if email_id not in inbox_ids:
            info["error"] = f"Email '{email_id}' not in inbox"
            return EmailReward(value=0.0, reason="unknown email id"), info

        # Duplicate action penalty
        if email_id in self._agent_labels:
            info["error"] = f"Email '{email_id}' already labeled"
            return EmailReward(value=LABEL_DUPLICATE, reason="duplicate label action"), info

        # Record label
        self._agent_labels[email_id] = label
        true_label = self._ground_truth_labels.get(email_id)

        if label == true_label:
            reward_val = LABEL_CORRECT
            reason = f"Correct: {email_id} = {label}"
        else:
            reward_val = LABEL_WRONG
            reason = f"Wrong: {email_id} got '{label}', expected '{true_label}'"

        # Episode done when all emails labeled
        if len(self._agent_labels) == len(self._inbox):
            self._done = True
            info["message"] = "All emails labeled — episode complete"

        return EmailReward(value=reward_val, reason=reason), info

    def _handle_rank(
        self, action_data: dict
    ) -> Tuple[EmailReward, Dict[str, Any]]:
        info: Dict[str, Any] = {"error": None}

        try:
            action = RankAction(**action_data)
        except Exception as e:
            info["error"] = f"Invalid rank action format: {e}"
            return EmailReward(value=0.0, reason="parse error"), info

        inbox_ids = {e.id for e in self._inbox}
        submitted_ids = set(action.order)

        # Warn if ids don't match but don't fail — grader handles partial
        missing = inbox_ids - submitted_ids
        extra = submitted_ids - inbox_ids
        if missing:
            info["warning"] = f"Missing email ids: {missing}"
        if extra:
            info["warning"] = f"Unknown email ids submitted: {extra}"

        # Score immediately — ranking is a single-action episode
        score, reason = score_ranking(action.order, self._ground_truth_ranking)
        self._done = True
        info["grader_score"] = score
        info["message"] = "Ranking submitted — episode complete"

        return EmailReward(value=score, reason=reason), info

    def _handle_reply(
        self, action_data: dict
    ) -> Tuple[EmailReward, Dict[str, Any]]:
        info: Dict[str, Any] = {"error": None}

        try:
            action = ReplyAction(**action_data)
        except Exception as e:
            info["error"] = f"Invalid reply action format: {e}"
            return EmailReward(value=0.0, reason="parse error"), info

        inbox_ids = {e.id for e in self._inbox}
        if action.email_id not in inbox_ids:
            info["error"] = f"Email '{action.email_id}' not in inbox"
            return EmailReward(value=0.0, reason="unknown email id"), info

        self._reply_attempts += 1
        max_attempts = 3

        # Score the reply
        score, reason = score_reply(action.body)
        penalty = REPLY_RETRY_PENALTY * (self._reply_attempts - 1)
        final_reward = max(0.0, round(score - penalty, 4))

        info["grader_score"] = score
        info["attempts"] = self._reply_attempts
        info["penalty"] = penalty

        if score >= 1.0 or self._reply_attempts >= max_attempts:
            self._done = True
            info["message"] = "Reply accepted — episode complete"
        else:
            info["message"] = (
                f"Reply scored {score:.2f}. "
                f"{max_attempts - self._reply_attempts} attempt(s) remaining. "
                f"Try to address all 3 issues."
            )

        return EmailReward(value=final_reward, reason=reason), info

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_observation(self) -> EmailObservation:
        """Build the observation object the agent will see."""
        instructions = TASK_CONFIG.get(self._task or "", {}).get("instructions", "")

        # Hide ground truth from agent — strip true_label and true_priority
        visible_emails = [
            Email(
                id=e.id,
                sender=e.sender,
                subject=e.subject,
                body=e.body,
                timestamp=e.timestamp,
                true_label=None,
                true_priority=None,
            )
            for e in self._inbox
        ]

        return EmailObservation(
            task=self._task or "",
            instructions=instructions,
            emails=visible_emails,
            actions_taken=copy.deepcopy(self._actions_taken),
            step=self._step_count,
            done=self._done,
        )

    def _expected_action(self) -> str:
        mapping = {
            "label": "label",
            "prioritize": "rank",
            "reply": "reply",
        }
        return mapping.get(self._task or "", "unknown")