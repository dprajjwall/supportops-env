"""
SupportOps-Env: Client
HTTP client for interacting with a running SupportOps-Env server.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import httpx

from models import SupportAction, SupportObservation, SupportState


class StepResult:
    """Result from a step or reset call."""
    def __init__(self, observation: SupportObservation, reward: Optional[float], done: bool):
        self.observation = observation
        self.reward = reward
        self.done = done

    def __repr__(self) -> str:
        return f"StepResult(reward={self.reward}, done={self.done})"


class SupportOpsEnv:
    """
    Synchronous HTTP client for SupportOps-Env.

    Usage:
        with SupportOpsEnv(base_url="http://localhost:8000") as env:
            result = env.reset(task_name="ticket_classification")
            print(result.observation.ticket)

            action = SupportAction(
                action_type="classify",
                payload={"category": "Bug"}
            )
            result = env.step(action)
            print(result.reward)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "SupportOpsEnv":
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> StepResult:
        """Reset the environment and start a new episode."""
        payload: Dict[str, Any] = {}
        if task_name:
            payload["task_name"] = task_name
        if seed is not None:
            payload["seed"] = seed
        if episode_id:
            payload["episode_id"] = episode_id

        resp = self._get_client().post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_result(data)

    def step(self, action: SupportAction) -> StepResult:
        """Execute an action and return the result."""
        resp = self._get_client().post(
            "/step",
            json={"action": action.model_dump()},
        )
        resp.raise_for_status()
        return self._parse_result(resp.json())

    def state(self) -> SupportState:
        """Get current episode state."""
        resp = self._get_client().get("/state")
        resp.raise_for_status()
        return SupportState(**resp.json())

    def health(self) -> Dict[str, str]:
        """Check server health."""
        resp = self._get_client().get("/health")
        resp.raise_for_status()
        return resp.json()

    def schema(self) -> Dict[str, Any]:
        """Get action/observation/state schemas."""
        resp = self._get_client().get("/schema")
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List available tasks."""
        resp = self._get_client().get("/tasks")
        resp.raise_for_status()
        return resp.json().get("tasks", [])

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    @staticmethod
    def _parse_result(data: Dict[str, Any]) -> StepResult:
        obs_data = data.get("observation", {})
        obs = SupportObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=data.get("reward"),
            done=data.get("done", False),
        )
