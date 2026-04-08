"""
SupportOps-Env: Pydantic Models
Action, Observation, State for the customer support triage environment.
"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class TicketData(BaseModel):
    """A customer support ticket."""
    model_config = ConfigDict(extra="allow")

    ticket_id: str
    subject: str
    body: str
    customer_tier: Literal["free", "pro", "enterprise"]
    created_at: str
    sentiment_score: float = 0.0      # -1.0 (angry) → +1.0 (happy)
    sla_hours: int = 48               # response SLA in hours
    # Ground-truth fields (hidden from agent in observations)
    category: Optional[str] = None
    priority: Optional[str] = None


class KBArticle(BaseModel):
    """A knowledge-base article."""
    article_id: str
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)



class SupportAction(BaseModel):
    """
    All actions the agent can take in SupportOps-Env.

    action_type options:
      - classify       : payload = {"category": str}
      - set_priority   : payload = {"ticket_id": str, "priority": str}
      - search_kb      : payload = {"query": str}
      - draft_response : payload = {"response_text": str}
      - mark_resolved  : payload = {"reason": str}
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    metadata: Dict[str, Any] = Field(default_factory=dict)
    action_type: Literal[
        "classify",
        "set_priority",
        "search_kb",
        "draft_response",
        "mark_resolved",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)



class SupportObservation(BaseModel):
    """
    Observation returned after reset() or step().
    Extends the base OpenEnv Observation fields.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # OpenEnv required base fields
    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Task context (always present)
    task_name: str = ""
    task_description: str = ""
    step_context: str = ""
    available_actions: List[str] = Field(default_factory=list)

    # Task 1 & 3: single ticket
    ticket: Optional[Dict[str, Any]] = None

    # Task 2: batch of tickets
    tickets: Optional[List[Dict[str, Any]]] = None

    # Task 3: KB search results and history
    kb_results: Optional[List[str]] = None
    customer_history: Optional[List[str]] = None

    # Running score feedback
    score_so_far: float = 0.0
    steps_remaining: int = 0


class SupportState(BaseModel):
    """Internal episode state."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # OpenEnv required base fields
    episode_id: Optional[str] = None
    step_count: int = 0

    # Environment-specific state
    task_name: str = ""
    current_ticket_id: Optional[str] = None
    max_steps: int = 10
    is_done: bool = False
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    cumulative_reward: float = 0.0
    priorities_set: Dict[str, str] = Field(default_factory=dict)  # Task 2
    kb_queries: List[str] = Field(default_factory=list)            # Task 3
    draft_submitted: bool = False                                   # Task 3
