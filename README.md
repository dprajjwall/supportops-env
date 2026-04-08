---
title: SupportOps-Env
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: bsd-3-clause
tags:
  - openenv
  - reinforcement-learning
  - rl-environment
  - nlp
  - customer-support
  - agent
short_description: OpenEnv RL environment for customer support triage
---

# 🎫 SupportOps-Env

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](Dockerfile)

> **An OpenEnv-compliant RL environment where AI agents learn to triage, classify, and respond to customer support tickets.**

Real-world utility: Every SaaS company operates a support queue. SupportOps-Env teaches agents to do the work of a Level-1 support team — intelligently routing tickets, prioritizing by urgency, and drafting accurate, empathetic responses using a knowledge base.

---

## 🌍 Environment Description

SupportOps-Env simulates a **customer support operations center**. The agent receives support tickets from a synthetic but realistic dataset of 60+ tickets spanning 5 categories, and must perform increasingly complex triage tasks.

### Why This Domain?

| Dimension | Details |
|-----------|---------|
| Real-world utility | Every B2B SaaS company has a support queue — this trains production-ready triage agents |
| Rich reward shaping | Partial credit at every step, not sparse binary rewards |
| Clear difficulty ladder | Easy → Medium → Hard across 3 tasks |
| Novel domain | Not represented in existing OpenEnv environments |
| Deterministic graders | Fully reproducible scores — no LLM-based evaluation |

---

## 📦 Action Space

All actions use the `SupportAction` model:

```json
{
  "action_type": "classify | set_priority | search_kb | draft_response | mark_resolved",
  "payload": { ... }
}
```

| `action_type` | Payload Fields | Used In |
|---------------|---------------|---------|
| `classify` | `category: str` | Task 1 |
| `set_priority` | `ticket_id: str`, `priority: str` | Task 2 |
| `search_kb` | `query: str` | Task 3 |
| `draft_response` | `response_text: str` | Task 3 |
| `mark_resolved` | `reason: str` | Task 3 |

Valid categories: `Bug`, `Feature Request`, `Billing`, `Account`, `General`

Valid priorities: `critical`, `high`, `medium`, `low`, `minimal`

---

## 👁️ Observation Space

All observations use the `SupportObservation` model:

```python
class SupportObservation(BaseModel):
    # OpenEnv required fields
    done: bool
    reward: Optional[float]
    metadata: Dict[str, Any]

    # Task context
    task_name: str
    task_description: str
    step_context: str
    available_actions: List[str]
    steps_remaining: int
    score_so_far: float

    # Task-specific fields
    ticket: Optional[Dict]        # Tasks 1 & 3: single ticket
    tickets: Optional[List[Dict]] # Task 2: 5 tickets to rank
    kb_results: Optional[List[str]]      # Task 3: KB search results
    customer_history: Optional[List[str]] # Task 3: customer history
```

Each ticket in the observation includes:

```json
{
  "ticket_id": "T001",
  "subject": "App crashes on login",
  "body": "Every time I try to log in...",
  "customer_tier": "enterprise",
  "created_at": "2024-01-15T09:00:00Z",
  "sentiment_score": -0.9,
  "sla_hours": 4
}
```

---

## 🎯 Tasks

### Task 1: Ticket Classification _(Easy)_

**Objective**: Read a support ticket and classify it into one of 5 categories.

**Max Steps**: 5  
**Difficulty**: Easy  
**Reward**: 1.0 (exact) | 0.5 (alias) | 0.0 (wrong)  

```json
{"action_type": "classify", "payload": {"category": "Bug"}}
```

**What makes it easy**: Single decision, immediate feedback, clear criteria.

---

### Task 2: Priority Queue Sorting _(Medium)_

**Objective**: Given 5 tickets, assign each a priority level (`critical`→`minimal`) reflecting true urgency.

**Max Steps**: 10  
**Difficulty**: Medium  
**Reward**: Composite score = 50% Kendall τ + 30% correct-critical-detection + 20% coverage  

```json
{"action_type": "set_priority", "payload": {"ticket_id": "T001", "priority": "critical"}}
```

**Priority guidelines** (must be inferred by agent):
- `critical`: system down, all users blocked, revenue at risk
- `high`: major feature broken, significant user impact
- `medium`: partial functionality broken, workaround exists
- `low`: cosmetic/minor issues, single user
- `minimal`: feature requests, questions

**What makes it medium**: Must compare tickets holistically, consider customer tier, SLA deadline, scope of impact.

---

### Task 3: Draft Response with KB Lookup _(Hard)_

**Objective**: Read a ticket, search the knowledge base, draft a professional response, and mark resolved.

**Max Steps**: 15  
**Difficulty**: Hard  
**Reward structure**:
- `+0.05` per relevant KB search query
- `+0.03` for submitting a first draft
- `−0.02` for off-topic KB queries
- `−0.10` for resolving without drafting
- **Final score** (35% resolution quality + 25% KB relevance + 25% tone + 15% efficiency)

```json
// Step 1: Search KB
{"action_type": "search_kb", "payload": {"query": "password reset email not received"}}

// Step 2: Draft response  
{"action_type": "draft_response", "payload": {"response_text": "Hi Sarah, I'm sorry to hear..."}}

// Step 3: Resolve
{"action_type": "mark_resolved", "payload": {"reason": "Password reset instructions provided"}}
```

**What makes it hard**: Multi-step reasoning, KB retrieval quality affects response quality, tone/empathy evaluation, efficiency pressure.

---

## 🏆 Reward Function Design

SupportOps-Env uses **dense rewards** (not sparse) to provide gradient signal throughout the episode:

```
Task 1: Immediate reward on classify action (1.0 / 0.5 / 0.0)

Task 2: No intermediate reward → final Kendall τ score when all 5 assigned
        (Encourages completing all assignments before scoring)

Task 3: +0.05 per relevant KB search   (reward exploration)
        -0.02 for off-topic search      (penalize wandering)
        +0.03 for first draft           (reward progress)
        -0.10 for resolve without draft (penalize shortcuts)
        Final: multi-component quality score
```

All rewards are clipped to `[0.0, 1.0]` at episode end.

---

## 🗂️ Dataset

- **60 synthetic tickets** across 5 categories (12 each)
- **5 priority sets** (pre-curated sets of 5 tickets for Task 2)
- **5 draft-response tasks** (paired ticket + expected KB topics)
- **20 KB articles** covering all ticket topics
- Ground-truth labels are hidden from the agent (server-side only)

---

## 🚀 Setup & Usage

### Prerequisites

```bash
pip install fastapi uvicorn pydantic httpx openai
```

### Run locally

```bash
# Start the server
cd c:/Prajjwal_Folder/Answer
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Verify
curl http://localhost:8000/health
# {"status": "healthy"}

# Reset (Task 1)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "ticket_classification", "seed": 42}'

# Step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "classify", "payload": {"category": "Bug"}}}'
```

### Docker

```bash
# Build
docker build -t supportops-env .

# Run
docker run -p 8000:8000 supportops-env

# Test
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

### Python client

```python
from client import SupportOpsEnv
from models import SupportAction

with SupportOpsEnv(base_url="http://localhost:8000") as env:
    # Task 1: Classification
    result = env.reset(task_name="ticket_classification", seed=42)
    print(result.observation.ticket)

    action = SupportAction(action_type="classify", payload={"category": "Bug"})
    result = env.step(action)
    print(f"Reward: {result.reward}, Done: {result.done}")
```

### Run baseline inference

```bash
# Set environment variables
export OPENAI_API_KEY="sk-..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export ENV_BASE_URL="http://localhost:8000"

# Run all 3 tasks
python inference.py
```

---

## 📊 Baseline Scores

Running `gpt-4o-mini` at temperature 0 (seed=42):

| Task | Difficulty | Expected Score | Notes |
|------|-----------|---------------|-------|
| ticket_classification | Easy | ~0.80 | Straightforward category matching |
| priority_sorting | Medium | ~0.55 | Requires multi-factor reasoning |
| draft_response | Hard | ~0.42 | KB search + tone + resolution quality |
| **Average** | — | **~0.59** | Above 0.5 success threshold |

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode (`task_name`, `seed`, `episode_id`) |
| `/step` | POST | Execute action, get observation + reward |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check (`{"status": "healthy"}`) |
| `/schema` | GET | JSON schemas for Action/Observation/State |
| `/tasks` | GET | List all tasks with descriptions |
| `/docs` | GET | Interactive Swagger UI |

---

## 📁 Project Structure

```
supportops-env/
├── openenv.yaml          # OpenEnv spec manifest
├── Dockerfile            # Container definition
├── inference.py          # Baseline inference script
├── models.py             # Pydantic models (Action, Observation, State)
├── client.py             # HTTP client
├── __init__.py
├── pyproject.toml
└── server/
    ├── app.py            # FastAPI application
    ├── environment.py    # Core environment logic
    ├── tasks.py          # Task registry and episode management
    ├── graders.py        # Deterministic scoring functions
    ├── tickets.py        # Ticket dataset (60 tickets + 20 KB articles)
    └── requirements.txt
```

---

## 🧪 Environment Validation

```bash
# Install validator
pip install openenv-core

# Run validation (requires live server)
openenv validate --url http://localhost:8000
```

Or use the pre-submission validation script:
```bash
./validate-submission.sh https://your-space.hf.space
```

---

## 📋 OpenEnv Compliance

| Requirement | Status |
|-------------|--------|
| Typed Pydantic models (Action, Observation, State) | ✅ |
| `POST /reset` → observation + reward + done | ✅ |
| `POST /step` → observation + reward + done | ✅ |
| `GET /state` → episode state | ✅ |
| `GET /health` → `{"status": "healthy"}` | ✅ |
| `GET /schema` → JSON schemas | ✅ |
| `openenv.yaml` with spec_version, name, type, runtime | ✅ |
| 3+ tasks with graders (0.0–1.0) | ✅ |
| Deterministic graders | ✅ |
| Baseline inference script at root | ✅ |
| [START]/[STEP]/[END] log format | ✅ |
| Working Dockerfile | ✅ |
| `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars | ✅ |
| OpenAI client for LLM calls | ✅ |

---

## 📜 License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built for the [Meta PyTorch OpenEnv Hackathon](https://github.com/meta-pytorch/OpenEnv).
Inspired by the Gymnasium API and the OpenEnv framework by Meta PyTorch.
