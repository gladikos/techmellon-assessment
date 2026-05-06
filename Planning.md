# TechMellon FDE Assessment — Notes

## Goal

Build an AI airline customer service agent that:
- handles customer requests,
- calls backend APIs,
- evaluates its own failures,
- improves automatically through iterations.

The focus is AI orchestration and autonomous refinement, not voice synthesis itself.

---

# 1. Knowledge Base

## Purpose
Store airline policies:
- baggage,
- pets,
- refunds,
- check-in,
- special assistance.

## Chosen Implementation
```text
knowledge_base.json
```

Why JSON: simple, lightweight, easy to load in Python, enough for static policy data.

Production Alternative: Move to a database, CMS, or admin-managed policy service.

---

# 2. Flight Database

## Purpose
Store:
- flights,
- bookings,
- seat availability,
- baggage additions,
- cancellations/reschedules.

## Chosen Database
```text
SQLite
```

Why SQLite: persistent, relational, no setup overhead, ideal for prototype transactional operations.

## Example Tables
- `flights`
- `bookings`
- `booking_addons`

---

# 3. FastAPI Webhooks / APIs

## Purpose
Allow the AI agent to interact with backend systems.

The agent should not hallucinate bookings, prices, availability, or baggage rules. It should call APIs.

## Planned Endpoints
```text
GET  /flights/search
POST /bookings
GET  /bookings/{reference}
POST /bookings/{reference}/cancel
POST /bookings/{reference}/reschedule
POST /bookings/{reference}/add-baggage
POST /bookings/{reference}/add-special-item
```

---

# 4. Autonomous Refinement Loop

## Goal
The system:
- simulates a customer call,
- evaluates the interaction,
- identifies failures,
- fixes prompt and/or code,
- reruns automatically.

## 4.1 Conversation Simulation

**Actors:**
- Customer Simulator LLM
- ElevenLabs Airline Agent

**Output:** Conversation transcript including messages, tool calls, API results.

## 4.2 Conversation Evaluation

Evaluator LLM checks:
- understanding,
- API correctness,
- conversational quality,
- outcome quality.

**Critical Output:** Root cause classification: `prompt` or `code`.

## 4.3 Automatic Fixing

- **Prompt Issue:** Rewrite system prompt.
- **Code Issue:** Generate targeted patch for specific backend function/file.

## 4.4 Update ElevenLabs Prompt

If prompt changes: push updated prompt to ElevenLabs API, rerun next iteration.

## 4.5 Termination

Loop stops when:
- all scores >= 8, or
- max iterations reached.

Example:
```text
PASS_THRESHOLD = 8
MAX_ITERATIONS = 5
```

---

# 5. Interface

## Purpose
Observation only.

Show:
- live transcript,
- evaluation scores,
- prompt diffs,
- iteration history,
- logs.

## Chosen UI
```text
Streamlit
```

---

# Planned LLM Usage

| Component          | Model                        |
|--------------------|------------------------------|
| Airline Agent      | ElevenLabs + Claude Sonnet   |
| Customer Simulator | Claude Haiku                 |
| Evaluator          | Claude Sonnet                |
| Prompt Fixer       | Claude Sonnet                |
| Code Fixer         | Claude Sonnet                |

---

# Engineering Philosophy

Focus on:
- pragmatic implementation,
- reliable integrations,
- controlled autonomous refinement,
- observability,
- fast iteration.

Avoid overengineering.
