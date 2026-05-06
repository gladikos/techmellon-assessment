# TechMellon Airline Agent — FDE Assessment

A self-improving voice AI agent for an airline customer service line, built on ElevenLabs Conversational AI. The system simulates customer calls, evaluates its own failures, classifies them as **prompt issues** or **code issues**, and autonomously rewrites prompts or patches backend code until all scenarios pass — or the iteration limit is reached.

> Forward Deployment Engineer technical assessment for TechMellon.

---

## Status

🚧 **Work in progress.** This README will be filled in as the project grows. Sections marked _TBD_ are placeholders.

---

## Architecture (high level)
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Customer         │◄────►│ ElevenLabs       │◄────►│ FastAPI Backend  │
│ Simulator (LLM)  │      │ Airline Agent    │      │ (bookings, KB)   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
▲                         ▲                         ▲
│                         │                         │
└────────┐    Transcript  │              ┌──────────┘
▼                ▼              ▼
┌─────────────────────────────────────┐
│ Evaluator → Prompt Fixer / Code Fixer│
│ (autonomous refinement loop)         │
└─────────────────────────────────────┘

---

## How to run

_TBD — will be filled in once the loop is wired up._

```powershell
# Clone and install
git clone <repo-url>
cd techmellon-assessment
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Configure secrets
Copy-Item .env.example .env
# Then fill in ANTHROPIC_API_KEY, ELEVENLABS_API_KEY, ELEVENLABS_AGENT_ID
```

---

## APIs and tools used

_TBD — final list will go here once integrations are complete._

- **FastAPI** — backend booking APIs and webhooks
- **SQLite** — flight and booking persistence
- **Anthropic Claude** — customer simulator, evaluator, prompt fixer, code fixer
- **ElevenLabs Conversational AI** — airline agent (Chat Mode for text-only conversations)
- **Streamlit** — observation UI

---

## Tradeoffs made

_TBD — will document decisions as they are made. A few committed up front:_

- **SQLite over Postgres.** Single-process, no setup, persists locally. Easier to demo. Would not survive concurrent writes at scale; sufficient for a single-process refinement loop.
- **JSON knowledge base over a vector store.** Policy data is small, static, and lookup-by-category. RAG would be over-engineering.
- **Raw `sqlite3` over SQLAlchemy.** Three tables, ~five query shapes; an ORM would be ceremony without payoff at this size.

---

## What I would improve next

_TBD — will be honest about gaps when complete._

---

## Project structure
techmellon-assessment/
├── backend/         # FastAPI server: booking + KB endpoints
├── agent/           # ElevenLabs agent config, prompt, tool schema
├── refinement/      # Autonomous loop: simulator, evaluator, fixers, orchestrator
├── ui/              # Streamlit observation UI
├── data/            # knowledge_base.json + airline.db (SQLite, generated)
└── logs/            # Structured JSONL run logs (generated)

---

## Recorded run

_TBD — will include a recorded example run showing starting prompt, scores per iteration, and the final refined prompt._