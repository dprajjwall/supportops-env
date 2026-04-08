"""
SupportOps-Env: Core Environment
Implements reset(), step(), state for all 3 tasks.
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import SupportAction, SupportObservation, SupportState
except ImportError:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from models import SupportAction, SupportObservation, SupportState  # type: ignore

try:
    from server.tasks import (
        TASK_CONFIGS,
        EpisodeState,
        create_episode,
        get_ticket_for_observation,
        get_tickets_for_observation,
    )
    from server.graders import (
        grade_classification,
        grade_priority_sorting,
        grade_draft_response,
        grade_draft_step_reward,
    )
    from server.tickets import TICKET_INDEX, search_kb
except ImportError:
    from tasks import (  # type: ignore
        TASK_CONFIGS,
        EpisodeState,
        create_episode,
        get_ticket_for_observation,
        get_tickets_for_observation,
    )
    from graders import (  # type: ignore
        grade_classification,
        grade_priority_sorting,
        grade_draft_response,
        grade_draft_step_reward,
    )
    from tickets import TICKET_INDEX, search_kb  # type: ignore


class SupportOpsEnvironment:
    """
    Customer Support Triage Environment.

    Tasks:
      - ticket_classification (easy)
      - priority_sorting (medium)
      - draft_response (hard)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._episode: Optional[EpisodeState] = None
        self._state: SupportState = SupportState()

    def reset(
        self,
        task_name: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """
        Start a new episode.

        Args:
            task_name: One of 'ticket_classification', 'priority_sorting', 'draft_response'.
                       Randomly chosen if not specified.
            seed: Random seed for reproducibility.
            episode_id: Optional custom episode ID.
        """
        import random
        if task_name is None:
            task_name = random.choice(list(TASK_CONFIGS.keys()))

        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from: {list(TASK_CONFIGS.keys())}"
            )

        self._episode = create_episode(task_name, seed=seed)
        config = TASK_CONFIGS[task_name]

        self._state = SupportState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            max_steps=config.max_steps,
            is_done=False,
            cumulative_reward=0.0,
            priorities_set={},
            kb_queries=[],
            draft_submitted=False,
        )

        return self._build_observation(done=False, reward=None, step_context="Episode started. Read the task and take action.")

    def step(
        self,
        action: SupportAction,
        **kwargs: Any,
    ) -> SupportObservation:
        """Execute one action in the current episode."""
        if self._episode is None or self._state.is_done:
            return SupportObservation(
                done=True,
                reward=0.0,
                task_name=self._state.task_name,
                step_context="Episode is not active. Call reset() first.",
                metadata={"error": "no_active_episode"},
            )

        self._state.step_count += 1
        ep = self._episode
        task = self._state.task_name

        reward = 0.0
        done = False
        step_context = ""

        # ── STEP LIMIT CHECK ──────────────────
        if self._state.step_count >= self._state.max_steps:
            done = True
            step_context = f"Step limit reached ({self._state.max_steps}). Episode ending."

            # Final scoring at step limit
            reward = self._compute_final_reward_at_limit()
            self._state.is_done = True
            self._state.cumulative_reward += reward

            obs = self._build_observation(done=True, reward=reward, step_context=step_context)
            obs.score_so_far = min(max(self._state.cumulative_reward, 0.01), 0.99)
            return obs

        # ── TASK 1: Ticket Classification ─────
        if task == "ticket_classification":
            reward, done, step_context = self._handle_classification(action, ep)

        # ── TASK 2: Priority Sorting ──────────
        elif task == "priority_sorting":
            reward, done, step_context = self._handle_priority(action, ep)

        # ── TASK 3: Draft Response ────────────
        elif task == "draft_response":
            reward, done, step_context = self._handle_draft(action, ep)

        else:
            step_context = f"Unknown task '{task}'."
            reward = 0.0
            done = True

        self._state.cumulative_reward += reward
        if done:
            self._state.is_done = True

        obs = self._build_observation(done=done, reward=reward, step_context=step_context)
        obs.score_so_far = min(max(self._state.cumulative_reward, 0.01), 0.99)
        return obs

    # ──────────────────────────────────────────
    # Task Handlers
    # ──────────────────────────────────────────
    def _handle_classification(
        self, action: SupportAction, ep: EpisodeState
    ):
        """Handle actions for Task 1."""
        if action.action_type != "classify":
            return (
                -0.05,
                False,
                f"Invalid action '{action.action_type}' for classification task. Use 'classify'.",
            )

        category = action.payload.get("category", "")
        ticket = TICKET_INDEX[ep.ticket_id]
        ground_truth = ticket.category

        score = grade_classification(category, ground_truth)
        if score == 1.0:
            context = f"[OK] Correct! Category '{category}' matches ground truth '{ground_truth}'."
        elif score > 0:
            context = (
                f"[~] Partial credit. '{category}' is close to the correct category '{ground_truth}'."
            )
        else:
            context = f"[X] Incorrect. '{category}' is not the right category."

        return score, True, context  # Always done after classification

    def _handle_priority(
        self, action: SupportAction, ep: EpisodeState
    ):
        """Handle actions for Task 2."""
        if action.action_type != "set_priority":
            return (
                -0.03,
                False,
                f"Invalid action '{action.action_type}'. Use 'set_priority'.",
            )

        ticket_id = action.payload.get("ticket_id", "")
        priority = action.payload.get("priority", "")

        if ticket_id not in ep.ticket_ids:
            return (-0.02, False, f"Unknown ticket_id '{ticket_id}'. Valid IDs: {ep.ticket_ids}")

        valid_priorities = {"critical", "high", "medium", "low", "minimal"}
        if priority.lower() not in valid_priorities:
            return (
                -0.02,
                False,
                f"Invalid priority '{priority}'. Choose from: {sorted(valid_priorities)}",
            )

        # Record assignment
        self._state.priorities_set[ticket_id] = priority.lower()
        ep.priorities_set[ticket_id] = priority.lower()

        all_assigned = all(tid in self._state.priorities_set for tid in ep.ticket_ids)
        context = (
            f"Priority set: ticket {ticket_id} → {priority}. "
            f"({len(self._state.priorities_set)}/{len(ep.ticket_ids)} tickets assigned)"
        )

        if all_assigned:
            # Compute final grade
            final_score = grade_priority_sorting(
                self._state.priorities_set,
                ep.ground_truth_order,
                ep.ticket_ids,
            )
            context += f" All tickets assigned. Final ranking score: {final_score:.2f}"
            return final_score, True, context

        else:
            return 0.0, False, context  # Intermediate step — no reward yet

    def _handle_draft(
        self, action: SupportAction, ep: EpisodeState
    ):
        """Handle actions for Task 3."""
        action_type = action.action_type

        if action_type == "search_kb":
            query = action.payload.get("query", "")
            if not query:
                return -0.02, False, "search_kb requires a 'query' in payload."

            ep.kb_queries.append(query)
            self._state.kb_queries.append(query)
            results = search_kb(query)

            step_reward = grade_draft_step_reward(
                action_type, action.payload,
                ep.kb_queries, ep.required_kb_tags,
                ep.draft_submitted,
            )
            context = (
                f"KB search results for '{query}':\n"
                + "\n\n".join(results[:2])
                + f"\n\n[Intermediate reward: {step_reward:+.2f}]"
            )
            return step_reward, False, context

        elif action_type == "draft_response":
            text = action.payload.get("response_text", "")
            if not text:
                return -0.02, False, "draft_response requires 'response_text' in payload."

            step_reward = grade_draft_step_reward(
                action_type, action.payload,
                ep.kb_queries, ep.required_kb_tags,
                ep.draft_submitted,
            )
            ep.draft_submitted = True
            self._state.draft_submitted = True
            ep.draft_text = text

            context = (
                f"Response drafted ({len(text.split())} words). "
                f"Now call 'mark_resolved' to finish the episode. "
                f"[Intermediate reward: {step_reward:+.2f}]"
            )
            return step_reward, False, context

        elif action_type == "mark_resolved":
            # Compute final grade
            final_score = grade_draft_response(
                draft_text=ep.draft_text,
                kb_queries=ep.kb_queries,
                expected_resolution_keywords=ep.expected_resolution_keywords,
                expected_tone_words=ep.expected_tone_words,
                required_kb_tags=ep.required_kb_tags,
                used_kb_articles=ep.used_kb_articles,
                step_count=self._state.step_count,
                max_steps=self._state.max_steps,
            )

            # Penalty if resolved without drafting
            if not ep.draft_submitted:
                final_score *= 0.1
                context = (
                    f"[X] Marked resolved without drafting a response. "
                    f"Final score: {final_score:.2f}"
                )
            else:
                context = f"[OK] Episode resolved. Final quality score: {final_score:.2f}"

            return final_score, True, context

        else:
            step_reward = grade_draft_step_reward(
                action_type, action.payload,
                ep.kb_queries, ep.required_kb_tags,
                ep.draft_submitted,
            )
            return step_reward, False, f"Unknown action '{action_type}' for this task."

    def _compute_final_reward_at_limit(self) -> float:
        """Compute partial reward when step limit is hit without explicit done."""
        ep = self._episode
        if ep is None:
            return 0.0
        task = self._state.task_name

        if task == "priority_sorting" and self._state.priorities_set:
            return grade_priority_sorting(
                self._state.priorities_set,
                ep.ground_truth_order,
                ep.ticket_ids,
            )
        if task == "draft_response" and ep.draft_text:
            return grade_draft_response(
                draft_text=ep.draft_text,
                kb_queries=ep.kb_queries,
                expected_resolution_keywords=ep.expected_resolution_keywords,
                expected_tone_words=ep.expected_tone_words,
                required_kb_tags=ep.required_kb_tags,
                used_kb_articles=ep.used_kb_articles,
                step_count=self._state.step_count,
                max_steps=self._state.max_steps,
            )
        return 0.0

    # ──────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────
    def _build_observation(
        self,
        done: bool,
        reward: Optional[float],
        step_context: str,
    ) -> SupportObservation:
        ep = self._episode
        if ep is None:
            return SupportObservation(
                done=done,
                reward=reward,
                task_name="",
                task_description="Call reset() to start.",
                step_context=step_context,
            )

        config = ep.config
        obs = SupportObservation(
            done=done,
            reward=reward,
            task_name=ep.task_name,
            task_description=config.description,
            step_context=step_context,
            available_actions=config.available_actions,
            steps_remaining=max(0, self._state.max_steps - self._state.step_count),
        )

        # Task 1 & 3: single ticket
        if ep.ticket_id:
            obs.ticket = get_ticket_for_observation(ep.ticket_id)

        # Task 2: multiple tickets
        if ep.ticket_ids:
            obs.tickets = get_tickets_for_observation(ep.ticket_ids)
            obs.ticket = None  # not needed for task 2

        # Task 3: KB results carry over from state
        if ep.task_name == "draft_response":
            obs.kb_results = []  # filled by search_kb action per step

        return obs

    @property
    def state(self) -> SupportState:
        return self._state

    def close(self) -> None:
        """Cleanup resources."""
        pass
