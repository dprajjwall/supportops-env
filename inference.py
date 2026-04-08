"""
SupportOps-Env: inference.py
Baseline inference script using OpenAI client against all 3 tasks.

Required environment variables:
  API_BASE_URL   - The API endpoint for the LLM (e.g., https://api.openai.com/v1)
  MODEL_NAME     - The model identifier (e.g., gpt-4o-mini)
  HF_TOKEN       - Your HuggingFace / API key (used as OPENAI_API_KEY if set)
  OPENAI_API_KEY - OpenAI API key (overrides HF_TOKEN)
  ENV_BASE_URL   - URL of the SupportOps-Env server (default: http://localhost:8000)

Output format (strict):
  [START] task=<name> max_steps=<n>
  [STEP]  step=<n> action=<json> reward=<float> done=<bool> error=<str|None>
  [END]   success=<bool> steps=<n> score=<float> rewards=<list>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import httpx
from openai import AsyncOpenAI

# ─────────────────────────────────────────────
# Configuration (from environment variables)
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")

# If HF_TOKEN is missing, look for OPENAI_API_KEY as a fallback
API_KEY = os.getenv("OPENAI_API_KEY", HF_TOKEN)


# Task parameters
TASKS = ["ticket_classification", "priority_sorting", "draft_response"]
MAX_STEPS: Dict[str, int] = {
    "ticket_classification": 5,
    "priority_sorting": 10,
    "draft_response": 15,
}
MAX_TOTAL_REWARD: Dict[str, float] = {
    "ticket_classification": 1.0,
    "priority_sorting": 1.0,
    "draft_response": 1.0,
}
SUCCESS_SCORE_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# Logging (strict format required by validator)
# ─────────────────────────────────────────────
def log_start(task: str, max_steps: int) -> None:
    print(f"[START] task={task} max_steps={max_steps}", flush=True)


def log_step(
    step: int,
    action: Any,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    action_str = json.dumps(action) if not isinstance(action, str) else action
    print(
        f"[STEP] step={step} action={action_str!r} "
        f"reward={reward} done={done} error={error}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    print(
        f"[END] success={success} steps={steps} "
        f"score={score:.4f} rewards={rewards}",
        flush=True,
    )


# ─────────────────────────────────────────────
# Environment HTTP helpers
# ─────────────────────────────────────────────
async def env_reset(
    client: httpx.AsyncClient,
    task_name: str,
    seed: int = 42,
) -> Dict[str, Any]:
    resp = await client.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_name": task_name, "seed": seed},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


async def env_step(
    client: httpx.AsyncClient,
    action: Dict[str, Any],
) -> Dict[str, Any]:
    resp = await client.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# LLM Prompt builders
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert customer support operations agent.
You interact with a support ticket environment by outputting structured JSON actions.
ALWAYS respond with ONLY a valid JSON object (no markdown, no code blocks, no extra text).

For ticket_classification tasks, respond with:
{"action_type": "classify", "payload": {"category": "Bug|Feature Request|Billing|Account|General"}}

For priority_sorting tasks, respond with:
{"action_type": "set_priority", "payload": {"ticket_id": "<ID>", "priority": "critical|high|medium|low|minimal"}}

For draft_response tasks, respond with ONE of:
{"action_type": "search_kb", "payload": {"query": "<relevant search query>"}}
{"action_type": "draft_response", "payload": {"response_text": "<your professional response>"}}
{"action_type": "mark_resolved", "payload": {"reason": "Response drafted and sent"}}

Rules:
- Always pick the most logical next action given the current observation
- For draft_response: search KB first, then draft a response, then mark_resolved
- For priority_sorting: assign each ticket one at a time with set_priority
- Output ONLY the JSON action, nothing else
"""


def build_user_message(obs: Dict[str, Any], history: List[str], task_name: str) -> str:
    """Build user message from current observation."""
    lines = [
        f"Task: {obs.get('task_name', task_name)}",
        f"Steps remaining: {obs.get('steps_remaining', '?')}",
        "",
    ]

    # Task description (first call only, via step_context)
    step_ctx = obs.get("step_context", "")
    if step_ctx:
        lines.append(f"Context: {step_ctx}")
        lines.append("")

    # Ticket(s)
    if obs.get("ticket"):
        t = obs["ticket"]
        lines += [
            "=== TICKET ===",
            f"ID: {t.get('ticket_id')}",
            f"Subject: {t.get('subject')}",
            f"Body: {t.get('body')}",
            f"Customer tier: {t.get('customer_tier')}",
            f"Sentiment: {t.get('sentiment_score', 0):.1f} (-1=angry, +1=happy)",
            f"SLA hours: {t.get('sla_hours')}",
            "==============",
        ]

    if obs.get("tickets"):
        lines.append("=== TICKETS TO PRIORITIZE ===")
        for t in obs["tickets"]:
            lines += [
                f"  [{t.get('ticket_id')}] {t.get('subject')}",
                f"    Tier: {t.get('customer_tier')} | Sentiment: {t.get('sentiment_score', 0):.1f} | SLA: {t.get('sla_hours')}h",
                f"    Body snippet: {t.get('body', '')[:120]}...",
                "",
            ]
        lines.append("=============================")

    # KB results
    if obs.get("kb_results"):
        lines.append("=== KB RESULTS ===")
        for r in obs["kb_results"]:
            lines.append(r[:300])
        lines.append("==================")

    # Recent history
    if history:
        lines.append("\nRecent history (last 5 steps):")
        for h in history[-5:]:
            lines.append(f"  {h}")

    # Task-specific instructions
    if task_name == "ticket_classification":
        lines.append("\nClassify this ticket now.")
    elif task_name == "priority_sorting":
        assigned = [h for h in history if "set_priority" in h]
        lines.append(f"\nAssigned so far: {len(assigned)} tickets. Assign the next one.")
    elif task_name == "draft_response":
        if not any("search_kb" in h for h in history):
            lines.append("\nStart by searching the knowledge base for relevant information.")
        elif not any("draft_response" in h for h in history):
            lines.append("\nNow draft your response to the customer.")
        else:
            lines.append("\nMark the ticket as resolved.")

    return "\n".join(lines)


def parse_action_from_llm(text: str, task_name: str) -> Dict[str, Any]:
    """
    Parse a JSON action from LLM output.
    Falls back to a default action if parsing fails.
    """
    # Try to extract JSON from the response
    text = text.strip()

    # Remove markdown code blocks if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if "action_type" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the text
    json_match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback defaults per task
    fallbacks = {
        "ticket_classification": {"action_type": "classify", "payload": {"category": "General"}},
        "priority_sorting": {"action_type": "set_priority", "payload": {"ticket_id": "T001", "priority": "medium"}},
        "draft_response": {"action_type": "mark_resolved", "payload": {"reason": "Unable to parse action"}},
    }
    return fallbacks.get(task_name, {"action_type": "mark_resolved", "payload": {"reason": "parse_error"}})


# ─────────────────────────────────────────────
# Main task runner
# ─────────────────────────────────────────────
async def run_task(
    openai_client: AsyncOpenAI,
    http_client: httpx.AsyncClient,
    task_name: str,
    seed: int = 42,
) -> Tuple[float, bool, List[float]]:
    """Run one task episode and return (score, success, rewards)."""
    max_steps = MAX_STEPS[task_name]
    max_total = MAX_TOTAL_REWARD[task_name]

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    history: List[str] = []

    log_start(task_name, max_steps)

    # Reset
    result = await env_reset(http_client, task_name, seed=seed)
    obs = result.get("observation", {})
    done = result.get("done", False)
    last_reward = 0.0

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        for step in range(1, max_steps + 1):
            if done:
                break

            # Build user message
            user_msg = build_user_message(obs, history, task_name)
            messages.append({"role": "user", "content": user_msg})

            # Get LLM action
            error = None
            action = {}
            try:
                completion = await openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=512,
                )
                llm_text = completion.choices[0].message.content or ""
                action = parse_action_from_llm(llm_text, task_name)
                messages.append({"role": "assistant", "content": llm_text})
            except Exception as e:
                error = str(e)
                action = {"action_type": "mark_resolved", "payload": {"reason": f"LLM error: {e}"}}

            # Step environment
            try:
                result = await env_step(http_client, action)
                obs = result.get("observation", {})
                reward = float(result.get("reward") or 0.0)
                done = result.get("done", False)
            except Exception as e:
                error = str(e)
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(
                f"Step {step}: {action.get('action_type')} "
                f"-> reward {reward:+.3f} done={done}"
            )

            if done:
                break

        # Score
        score = sum(rewards) / max_total if max_total > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, success, rewards


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
async def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or OPENAI_API_KEY environment variable.", flush=True)
        sys.exit(1)

    openai_client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

    all_scores: Dict[str, float] = {}

    async with httpx.AsyncClient() as http_client:
        # Verify server is running
        try:
            resp = await http_client.get(f"{ENV_BASE_URL}/health", timeout=10.0)
            resp.raise_for_status()
            print(f"[INFO] Connected to SupportOps-Env at {ENV_BASE_URL}", flush=True)
        except Exception as e:
            print(f"[ERROR] Cannot connect to environment server at {ENV_BASE_URL}: {e}", flush=True)
            print("[INFO] Start the server with: uvicorn server.app:app --port 8000", flush=True)
            sys.exit(1)

        for task_name in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"Running task: {task_name}", flush=True)
            print(f"{'='*60}", flush=True)
            score, success, rewards = await run_task(
                openai_client, http_client, task_name, seed=42
            )
            all_scores[task_name] = score

    # Final summary
    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for task, s in all_scores.items():
        status = "✓ PASS" if s >= SUCCESS_SCORE_THRESHOLD else "✗ FAIL"
        print(f"  {task:<30} score={s:.4f}  {status}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"\n  {'AVERAGE':<30} score={avg:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
