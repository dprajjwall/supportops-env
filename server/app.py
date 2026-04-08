"""
SupportOps-Env: FastAPI Server Application
Exposes OpenEnv-compliant HTTP endpoints:
  POST /reset    - Reset environment, returns initial observation
  POST /step     - Execute action, returns observation + reward + done
  GET  /state    - Returns current episode state
  GET  /health   - Health check
  GET  /schema   - Action/Observation/State JSON schemas
  GET  /tasks    - List available tasks with descriptions
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from models import SupportAction, SupportObservation, SupportState
    from server.environment import SupportOpsEnvironment
    from server.tasks import TASK_CONFIGS
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from models import SupportAction, SupportObservation, SupportState  # type: ignore
    from environment import SupportOpsEnvironment  # type: ignore
    from tasks import TASK_CONFIGS  # type: ignore

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="SupportOps-Env",
    description=(
        "An OpenEnv-compliant RL environment for training agents to triage, "
        "classify, and respond to customer support tickets."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session; for multi-session use WebSocket pattern)
_env = SupportOpsEnvironment()


class ResetRequest(BaseModel):
    model_config = {"extra": "allow"}
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: Optional[str] = Field(
        default=None,
        description="Task to run: ticket_classification | priority_sorting | draft_response",
    )


class StepRequest(BaseModel):
    model_config = {"extra": "allow"}
    action: Dict[str, Any] = Field(
        ...,
        description="Action dict matching SupportAction schema",
        examples=[{"action_type": "classify", "payload": {"category": "Bug"}}],
    )


class EnvResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float]
    done: bool


def _obs_to_response(obs: SupportObservation) -> EnvResponse:
    return EnvResponse(
        observation=obs.model_dump(),
        reward=obs.reward,
        done=obs.done,
    )


@app.post(
    "/reset",
    response_model=EnvResponse,
    summary="Reset the environment",
    description="Start a new episode. Optionally specify task_name and seed.",
)
async def reset(request: ResetRequest = ResetRequest()) -> EnvResponse:
    try:
        obs = _env.reset(
            task_name=request.task_name,
            seed=request.seed,
            episode_id=request.episode_id,
        )
        return _obs_to_response(obs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post(
    "/step",
    response_model=EnvResponse,
    summary="Execute an action",
    description="Send an action and receive the resulting observation, reward, and done flag.",
)
async def step(request: StepRequest = None) -> EnvResponse:
    if request is None or not request.action:
        raise HTTPException(status_code=400, detail="Action is required in request body")
    try:
        action = SupportAction(**request.action)
        obs = _env.step(action)
        return _obs_to_response(obs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid action or step failed: {e}")



@app.get(
    "/state",
    summary="Get current episode state",
    description="Returns internal state: episode_id, step_count, task_name, etc.",
)
async def state() -> Dict[str, Any]:
    return _env.state.model_dump()


@app.get(
    "/health",
    summary="Health check",
    description="Returns server health status.",
)
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get(
    "/schema",
    summary="Get Action/Observation/State schemas",
    description="Returns JSON schemas for action, observation, and state models.",
)
async def schema() -> Dict[str, Any]:
    return {
        "action": SupportAction.model_json_schema(),
        "observation": SupportObservation.model_json_schema(),
        "state": SupportState.model_json_schema(),
    }


@app.get(
    "/tasks",
    summary="List available tasks",
    description="Returns all 3 tasks with descriptions and difficulty levels.",
)
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": cfg.name,
                "difficulty": cfg.difficulty,
                "max_steps": cfg.max_steps,
                "description": cfg.description[:200] + "...",
                "available_actions": cfg.available_actions,
            }
            for cfg in TASK_CONFIGS.values()
        ]
    }


@app.get("/", summary="Root")
async def root() -> Dict[str, str]:
    return {
        "name": "SupportOps-Env",
        "version": "1.0.0",
        "description": "OpenEnv-compliant customer support triage environment",
        "tasks": "ticket_classification | priority_sorting | draft_response",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
