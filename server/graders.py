"""
SupportOps-Env: Graders
Deterministic scoring functions for each of the 3 tasks.
All graders return a float in [0.0, 1.0].
"""
from __future__ import annotations
from typing import Any, Dict, List

def _clip_score(score: float) -> float:
    """Clips the score strictly within (0, 1) interval to pass Phase 2 validation."""
    return min(max(score, 0.01), 0.99)

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 Grader: Ticket Classification
# ─────────────────────────────────────────────────────────────────────────────
VALID_CATEGORIES = {"Bug", "Feature Request", "Billing", "Account", "General"}

# Aliases that map common agent phrasings to canonical categories
CATEGORY_ALIASES: Dict[str, str] = {
    # Bug aliases
    "bug": "Bug",
    "bugs": "Bug",
    "defect": "Bug",
    "error": "Bug",
    "issue": "Bug",
    "crash": "Bug",
    "technical issue": "Bug",
    # Feature Request aliases
    "feature request": "Feature Request",
    "feature": "Feature Request",
    "enhancement": "Feature Request",
    "request": "Feature Request",
    "improvement": "Feature Request",
    "suggestion": "Feature Request",
    # Billing aliases
    "billing": "Billing",
    "payment": "Billing",
    "invoice": "Billing",
    "charge": "Billing",
    "subscription": "Billing",
    "refund": "Billing",
    # Account aliases
    "account": "Account",
    "login": "Account",
    "access": "Account",
    "password": "Account",
    "user": "Account",
    "profile": "Account",
    "security": "Account",
    # General
    "general": "General",
    "information": "General",
    "question": "General",
    "inquiry": "General",
    "other": "General",
}

def grade_classification(predicted: str, ground_truth: str) -> float:
    """
    Score a single ticket classification.

    Returns:
        1.0  - exact match (case-insensitive)
        0.5  - alias/variant maps to correct category
        0.0  - wrong category
    """
    if not predicted:
        return _clip_score(0.0)

    # Normalize
    pred_clean = predicted.strip()

    # Exact match (case-insensitive)
    if pred_clean.lower() == ground_truth.lower():
        return _clip_score(1.0)

    # Check if it's a known alias
    canonical = CATEGORY_ALIASES.get(pred_clean.lower())
    if canonical and canonical.lower() == ground_truth.lower():
        return _clip_score(0.5)

    # Check if the ground truth is contained in the prediction
    if ground_truth.lower() in pred_clean.lower():
        return _clip_score(0.5)

    return _clip_score(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 Grader: Priority Queue Sorting
# ─────────────────────────────────────────────────────────────────────────────
PRIORITY_LEVELS = {"critical": 5, "high": 4, "medium": 3, "low": 2, "minimal": 1}

def _kendall_tau_score(pred_order: List[str], truth_order: List[str]) -> float:
    """
    Compute normalized Kendall tau correlation between two orderings.
    Returns a value in [-1, 1], then scaled to [0, 1].
    Both lists must contain the same ticket IDs.
    """
    n = len(pred_order)
    if n <= 1:
        return 1.0

    # Build position maps
    pred_pos = {tid: i for i, tid in enumerate(pred_order)}
    truth_pos = {tid: i for i, tid in enumerate(truth_order)}

    concordant = 0
    discordant = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = truth_order[i], truth_order[j]
            # In truth: ti comes before tj (ti is more urgent)
            # Check if pred agrees
            if ti in pred_pos and tj in pred_pos:
                if pred_pos[ti] < pred_pos[tj]:
                    concordant += 1
                else:
                    discordant += 1

    if total_pairs == 0:
        return 1.0

    tau = (concordant - discordant) / total_pairs  # in [-1, 1]
    # Scale to [0, 1]
    return (tau + 1.0) / 2.0


def grade_priority_sorting(
    priorities_set: Dict[str, str],   # ticket_id -> priority_label set by agent
    ground_truth_order: List[str],    # ticket IDs from most to least urgent
    ticket_ids: List[str],
) -> float:
    """
    Grade the priority sorting task.

    Returns a composite score:
    - 60% Kendall tau on agent-provided ordering
    - 40% correct critical/non-critical classification
    """
    if not priorities_set:
        return _clip_score(0.0)

    # Build agent's ordering from assigned priority levels
    def priority_value(label: str) -> int:
        return PRIORITY_LEVELS.get(label.strip().lower(), 0)

    # Sort agent's assignments from highest to lowest priority
    sorted_by_agent = sorted(
        priorities_set.keys(),
        key=lambda tid: priority_value(priorities_set[tid]),
        reverse=True,
    )

    # Only score tickets that appear in both ground truth and agent assignments
    valid_truth = [t for t in ground_truth_order if t in priorities_set]
    valid_pred = [t for t in sorted_by_agent if t in ground_truth_order]

    if not valid_truth or not valid_pred:
        return _clip_score(0.0)

    # Kendall tau component (60%)
    tau_score = _kendall_tau_score(valid_pred, valid_truth)

    # Critical detection: does agent correctly identify the most urgent ticket?
    most_urgent_gt = ground_truth_order[0]
    most_urgent_pred = sorted_by_agent[0] if sorted_by_agent else ""
    critical_match = 1.0 if most_urgent_pred == most_urgent_gt else 0.0

    # Coverage bonus: reward for classifying all 5 tickets
    coverage = len(priorities_set) / max(len(ticket_ids), 1)

    # Composite
    score = 0.50 * tau_score + 0.30 * critical_match + 0.20 * coverage
    return _clip_score(score)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 Grader: Draft Response with KB lookup
# ─────────────────────────────────────────────────────────────────────────────
def grade_draft_response(
    draft_text: str,
    kb_queries: List[str],
    expected_resolution_keywords: List[str],
    expected_tone_words: List[str],
    required_kb_tags: List[str],
    used_kb_articles: List[str],     # article IDs retrieved by agent
    step_count: int,
    max_steps: int,
) -> float:
    """
    Grade a drafted response for Task 3.

    Components (max 1.0 total):
    - 0.25  KB relevance (did agent search appropriate topics?)
    - 0.35  Resolution quality (are expected keywords in the response?)
    - 0.25  Tone quality (empathy, professionalism)
    - 0.15  Efficiency (fewer steps = better, up to a point)
    """
    if not draft_text:
        return _clip_score(0.0)

    draft_lower = draft_text.lower()
    score = 0.0

    # ── KB Relevance (0.25) ───────────────────
    kb_score = 0.0
    if kb_queries:
        matched_tags = 0
        for q in kb_queries:
            q_lower = q.lower()
            for tag in required_kb_tags:
                tag_words = tag.replace("-", " ").split()
                if any(w in q_lower for w in tag_words):
                    matched_tags += 1
                    break
        kb_score = min(matched_tags / max(len(required_kb_tags), 1), 1.0)
    # Bonus if agent retrieved relevant article IDs
    if used_kb_articles:
        kb_score = min(kb_score + 0.15, 1.0)
    score += 0.25 * kb_score

    # ── Resolution Quality (0.35) ─────────────
    if expected_resolution_keywords:
        hits = sum(
            1 for kw in expected_resolution_keywords if kw.lower() in draft_lower
        )
        resolution_score = hits / len(expected_resolution_keywords)
    else:
        resolution_score = 0.5
    score += 0.35 * resolution_score

    # ── Tone Quality (0.25) ───────────────────
    if expected_tone_words:
        tone_hits = sum(
            1 for tw in expected_tone_words if tw.lower() in draft_lower
        )
        tone_score = min(tone_hits / max(len(expected_tone_words) * 0.6, 1), 1.0)
    else:
        tone_score = 0.5
    score += 0.25 * tone_score

    # ── Efficiency (0.15) ─────────────────────
    # Full score if done in ≤ 60% of max_steps; 0 if used all steps
    if max_steps > 0:
        step_ratio = step_count / max_steps
        efficiency = max(0.0, 1.0 - step_ratio)
    else:
        efficiency = 0.0
    score += 0.15 * efficiency

    return _clip_score(score)


def grade_draft_step_reward(
    action_type: str,
    payload: Dict[str, Any],
    kb_queries: List[str],
    required_kb_tags: List[str],
    draft_submitted: bool,
) -> float:
    """
    Per-step reward shaping for Task 3 (intermediate rewards).

    Returns a small reward or penalty for each action taken.
    """
    if action_type == "search_kb":
        query = payload.get("query", "").lower()
        # Reward relevant queries (tags appear in query)
        relevant = any(
            any(w in query for w in tag.replace("-", " ").split())
            for tag in required_kb_tags
        )
        if relevant:
            return 0.05   # small reward for relevant search
        else:
            return -0.02  # small penalty for off-topic search

    if action_type == "draft_response":
        if not draft_submitted:
            return 0.03  # first draft attempt
        else:
            return -0.02  # penalize multiple draft attempts

    if action_type == "mark_resolved":
        if draft_submitted:
            return 0.0  # neutral — final score from grader
        else:
            return -0.10  # resolved without drafting!

    return 0.0  # no-op reward for unknown actions
