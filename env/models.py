from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ── Core domain object ──────────────────────────────────────────────

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    true_label: Optional[Literal["spam", "urgent", "fyi", "action-needed"]] = None
    true_priority: Optional[int] = None


# ── What the agent sees ─────────────────────────────────────────────

class EmailObservation(BaseModel):
    task: Literal["label", "prioritize", "reply"]
    instructions: str
    emails: List[Email]
    actions_taken: List[dict] = Field(default_factory=list)
    step: int = 0
    done: bool = False


# ── Actions the agent can send ──────────────────────────────────────

class LabelAction(BaseModel):
    action: Literal["label"]
    email_id: str
    label: Literal["spam", "urgent", "fyi", "action-needed"]


class RankAction(BaseModel):
    action: Literal["rank"]
    order: List[str]


class ReplyAction(BaseModel):
    action: Literal["reply"]
    email_id: str
    body: str


EmailAction = LabelAction | RankAction | ReplyAction


# ── Reward and state ────────────────────────────────────────────────

class EmailReward(BaseModel):
    value: float = 0.0
    reason: str = ""


class EnvState(BaseModel):
    task: str
    step: int
    done: bool
    inbox: List[Email]
    actions_taken: List[dict]
    total_reward: float