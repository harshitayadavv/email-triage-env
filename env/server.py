from __future__ import annotations
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import EmailTriageEnv


app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv-compatible RL environment for email triage",
    version="1.0.0",
)

env = EmailTriageEnv()


# ── Request bodies ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "label"


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "email-triage-env",
        "version": "1.0.0",
        "tasks": ["label", "prioritize", "reply"],
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    task = request.task if request else "label"
    try:
        obs = env.reset(task)
        return {
            "observation": obs.model_dump(),
            "info": {},
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(request: StepRequest):
    try:
        obs, reward, done, info = env.step(request.action)

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
    try:
        s = env.state()
        return {"state": s.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score")
def score():
    try:
        final_score, reason = env.final_score()
        return {
            "score": final_score,
            "reason": reason,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("env.server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()