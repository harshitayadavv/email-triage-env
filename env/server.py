from __future__ import annotations
import os
from typing import Any, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import EmailTriageEnv


app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv-compatible RL environment for email triage",
    version="1.0.0",
)

# One global env instance per server process
env = EmailTriageEnv()


# ── Request bodies ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "label"


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Ping endpoint — judges check this first."""
    return {"status": "ok"}


@app.get("/")
def root():
    """Root endpoint — also returns 200 for HF Space ping."""
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "tasks": ["label", "prioritize", "reply"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.post("/reset")
def reset(request: ResetRequest):
    """
    Reset the environment for a given task.
    Body: {"task": "label" | "prioritize" | "reply"}
    Returns: initial observation
    """
    try:
        obs = env.reset(request.task)
        return {
            "observation": obs.model_dump(),
            "info": {},
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    """
    Take one action in the environment.
    Body: {"action": {...}}
    Returns: observation, reward, done, info
    """
    try:
        obs, reward, done, info = env.step(request.action)

        # If episode just ended, attach final grader score
        if done:
            final_score, final_reason = env.final_score()
            info["final_score"] = final_score
            info["final_reason"] = final_reason

        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/state")
def state():
    """
    Return full internal environment state.
    No body required.
    """
    try:
        s = env.state()
        return {"state": s.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score")
def score():
    """
    Run the grader and return the final episode score.
    Call after done=True.
    """
    try:
        final_score, reason = env.final_score()
        return {
            "score": final_score,
            "reason": reason,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("env.server:app", host="0.0.0.0", port=port, reload=False)