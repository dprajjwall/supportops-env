"""
SupportOps-Env: Task Registry
Loads and manages the 3 task definitions and their configurations.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from server.tickets import (
        TICKETS,
        TICKET_INDEX,
        PRIORITY_SETS,
        DRAFT_TASKS,
        search_kb,
    )
except ImportError:
    from tickets import (  # type: ignore
        TICKETS,
        TICKET_INDEX,
        PRIORITY_SETS,
        DRAFT_TASKS,
        search_kb,
    )


# ─────────────────────────────────────────────
# Task Config
# ─────────────────────────────────────────────
@dataclass
class TaskConfig:
    name: str
    description: str
    max_steps: int
    available_actions: List[str]
    difficulty: str


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "ticket_classification": TaskConfig(
        name="ticket_classification",
        difficulty="easy",
        max_steps=5,
        description=(
            "Read the customer support ticket provided and classify it into exactly ONE "
            "of the following categories: Bug, Feature Request, Billing, Account, General.\n\n"
            "A 'Bug' is something that is broken or not working as expected.\n"
            "A 'Feature Request' is a suggestion for new functionality.\n"
            "'Billing' relates to charges, invoices, payments, or refunds.\n"
            "'Account' relates to login, access, users, or security.\n"
            "'General' is for questions, inquiries, or information requests.\n\n"
            "To classify, call the 'classify' action with:\n"
            '  {"action_type": "classify", "payload": {"category": "<CATEGORY>"}}\n\n'
            "You will receive a reward of 1.0 for a correct classification, "
            "0.5 for a close match, and 0.0 for incorrect."
        ),
        available_actions=[
            'classify: {"action_type": "classify", "payload": {"category": "Bug|Feature Request|Billing|Account|General"}}'
        ],
    ),
    "priority_sorting": TaskConfig(
        name="priority_sorting",
        difficulty="medium",
        max_steps=10,
        description=(
            "You will receive 5 customer support tickets. "
            "Assign each ticket a priority level: critical, high, medium, low, or minimal.\n\n"
            "Priority guidelines:\n"
            "- critical: system down, data loss, complete lockout, all users blocked\n"
            "- high: major feature broken, significant revenue impact, many users affected\n"
            "- medium: partial functionality broken, workaround exists, billing discrepancies\n"
            "- low: minor bugs, single user issues, questions, cosmetic issues\n"
            "- minimal: feature requests, typos, nice-to-have improvements\n\n"
            "Consider: customer tier (enterprise > pro > free), sentiment, SLA hours, "
            "number of users affected.\n\n"
            "To set priority for each ticket, call:\n"
            '  {"action_type": "set_priority", "payload": {"ticket_id": "<ID>", '
            '"priority": "critical|high|medium|low|minimal"}}\n\n'
            "Call this once per ticket. You get partial credit based on how well your "
            "ranking correlates with the ground-truth urgency ordering."
        ),
        available_actions=[
            'set_priority: {"action_type": "set_priority", "payload": {"ticket_id": "<ID>", "priority": "critical|high|medium|low|minimal"}}'
        ],
    ),
    "draft_response": TaskConfig(
        name="draft_response",
        difficulty="hard",
        max_steps=15,
        description=(
            "You are a support agent. Read the ticket, search the knowledge base for "
            "relevant information, then draft a professional response to the customer.\n\n"
            "Steps:\n"
            "1. Search the KB with relevant queries to find helpful articles\n"
            "2. Draft a clear, professional, empathetic response that addresses the issue\n"
            "3. Mark the ticket as resolved once your response is ready\n\n"
            "Available actions:\n"
            '  search_kb: {"action_type": "search_kb", "payload": {"query": "<SEARCH QUERY>"}}\n'
            '  draft_response: {"action_type": "draft_response", "payload": {"response_text": "<YOUR RESPONSE>"}}\n'
            '  mark_resolved: {"action_type": "mark_resolved", "payload": {"reason": "<REASON>"}}\n\n'
            "Scoring: You earn rewards for relevant KB searches (+0.05), drafting a response (+0.03), "
            "and addressing the issue correctly. Penalized for off-topic searches (-0.02) and "
            "resolving without drafting (-0.10). Final score based on resolution quality."
        ),
        available_actions=[
            'search_kb: {"action_type": "search_kb", "payload": {"query": "<SEARCH QUERY>"}}',
            'draft_response: {"action_type": "draft_response", "payload": {"response_text": "<RESPONSE>"}}',
            'mark_resolved: {"action_type": "mark_resolved", "payload": {"reason": "<REASON>"}}',
        ],
    ),
}


# ─────────────────────────────────────────────
# Episode State (per task type)
# ─────────────────────────────────────────────
@dataclass
class EpisodeState:
    task_name: str
    config: TaskConfig

    # Task 1
    ticket_id: Optional[str] = None

    # Task 2
    priority_set_id: Optional[str] = None
    ticket_ids: List[str] = field(default_factory=list)
    ground_truth_order: List[str] = field(default_factory=list)
    priorities_set: Dict[str, str] = field(default_factory=dict)

    # Task 3
    draft_task_id: Optional[str] = None
    required_kb_tags: List[str] = field(default_factory=list)
    expected_resolution_keywords: List[str] = field(default_factory=list)
    expected_tone_words: List[str] = field(default_factory=list)
    kb_queries: List[str] = field(default_factory=list)
    used_kb_articles: List[str] = field(default_factory=list)
    draft_submitted: bool = False
    draft_text: str = ""


def create_episode(task_name: str, seed: Optional[int] = None) -> EpisodeState:
    """
    Initialize a new episode for the given task.
    Randomly selects ticket/set from the dataset.
    """
    rng = random.Random(seed)
    config = TASK_CONFIGS[task_name]
    state = EpisodeState(task_name=task_name, config=config)

    if task_name == "ticket_classification":
        ticket = rng.choice(TICKETS)
        state.ticket_id = ticket.ticket_id

    elif task_name == "priority_sorting":
        pset = rng.choice(PRIORITY_SETS)
        state.priority_set_id = pset["set_id"]
        state.ticket_ids = list(pset["ticket_ids"])
        state.ground_truth_order = list(pset["ground_truth_order"])

    elif task_name == "draft_response":
        dt = rng.choice(DRAFT_TASKS)
        state.draft_task_id = dt["task_id"]
        state.ticket_id = dt["ticket_id"]
        state.required_kb_tags = list(dt["required_kb_tags"])
        state.expected_resolution_keywords = list(dt["expected_resolution_keywords"])
        state.expected_tone_words = list(dt["expected_tone_words"])

    return state


def get_ticket_for_observation(ticket_id: str) -> Dict[str, Any]:
    """Return a ticket dict safe for agent observation (no ground-truth labels)."""
    t = TICKET_INDEX[ticket_id]
    return {
        "ticket_id": t.ticket_id,
        "subject": t.subject,
        "body": t.body,
        "customer_tier": t.customer_tier,
        "created_at": t.created_at,
        "sentiment_score": t.sentiment_score,
        "sla_hours": t.sla_hours,
    }


def get_tickets_for_observation(ticket_ids: List[str]) -> List[Dict[str, Any]]:
    """Return multiple tickets (no ground-truth labels)."""
    return [get_ticket_for_observation(tid) for tid in ticket_ids]
