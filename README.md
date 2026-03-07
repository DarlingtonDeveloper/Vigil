# Vigil

**Agentic risk pricing engine for AI agent deployments.**

Vigil is a multi-agent system that assesses the risk profile of autonomous AI deployments. It analyses legal exposure, technical vulnerabilities, and mitigation posture across jurisdictions, then produces a priced risk score with actionable recommendations. Built with LangGraph, Claude, and a SurrealDB knowledge graph.

---

## How It Works

Every assessment runs through a **6-stage pipeline** orchestrated by LangGraph:

```
Intake --> Knowledge Fetch --> Legal --> Technical --> Mitigation --> Pricing
```

| Stage | Agent | What it does |
|-------|-------|-------------|
| 1. **Intake** | `intake_node` | Parses the deployment description into a structured profile — tools, data access levels, autonomy, output reach, oversight, guardrails |
| 2. **Knowledge Fetch** | `fetch_knowledge_node` | Queries the SurrealDB knowledge graph for applicable doctrines, regulations, risk factors, and mitigations (no LLM call) |
| 3. **Legal** | `legal_node` | Assesses legal exposure — doctrine applicability, regulatory gaps, and an exposure score per jurisdiction |
| 4. **Technical** | `technical_node` | Scores each risk factor from the taxonomy, identifies amplification effects between factors |
| 5. **Mitigation** | `mitigation_node` | Evaluates current controls across four axes (legal conformity, human oversight, architectural, evidentiary) and recommends improvements |
| 6. **Pricing** | `pricing_node` | Synthesises all analyses into an overall risk score, premium band, scenarios, and prioritised recommendations |

Each LLM stage is validated by an **Opik quality gate** before results propagate downstream.

---

## Architecture

```
                        +-----------------+
                        |   Next.js App   |  AI SDK chat + risk dashboard
                        +--------+--------+
                                 |
                        +--------v--------+
                        |   FastAPI        |  REST API + middleware stack
                        |  (auth, rate     |  (auth, validation, rate limit,
                        |   limit, audit)  |   audit logging)
                        +--------+--------+
                                 |
                        +--------v--------+
                        |   LangGraph      |  6-node StateGraph pipeline
                        |   Workflow       |
                        +--------+--------+
                                 |
                   +-------------+-------------+
                   |             |             |
            +------v---+  +-----v----+  +-----v------+
            |  Claude   |  | SurrealDB|  | Opik       |
            |  Sonnet   |  | Knowledge|  | Quality    |
            |  (LLM)    |  | Graph    |  | Gates      |
            +-----------+  +----------+  +------------+
```

---

## Tech Stack

### Backend

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI 0.115 |
| LLM orchestration | LangGraph 0.2 + LangChain 0.3 |
| LLM | Claude Sonnet (via langchain-anthropic) |
| Database | SurrealDB 2.2 (graph + document) |
| Observability | Opik (quality gates) + LangSmith (tracing) |
| Cost tracking | Custom per-session token/USD tracker |
| Validation | Pydantic v2 |

### Frontend

| Component | Technology |
|-----------|-----------|
| Framework | Next.js 16 (App Router) |
| UI | React 19 + Tailwind CSS 4 |
| AI chat | Vercel AI SDK (@ai-sdk/anthropic) |
| Graph viz | react-force-graph-2d |
| Schema validation | Zod 4 |
| Testing | Vitest + Testing Library |

---

## Project Structure

```
vigil/
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pytest.ini
│   └── app/
│       ├── main.py                  # FastAPI entry, lifespan startup
│       ├── config.py                # Pydantic settings
│       ├── agents/
│       │   ├── intake.py            # Stage 1 — deployment profiling
│       │   ├── legal.py             # Stage 3 — legal exposure analysis
│       │   ├── technical.py         # Stage 4 — technical risk scoring
│       │   ├── mitigation.py        # Stage 5 — control evaluation
│       │   └── pricing.py           # Stage 6 — risk pricing
│       ├── graph/
│       │   ├── state.py             # VigilState TypedDict
│       │   └── workflow.py          # LangGraph StateGraph (6 nodes)
│       ├── routes/
│       │   ├── assess.py            # POST /api/assess
│       │   ├── scenario.py          # POST /api/scenario
│       │   ├── knowledge.py         # GET  /api/knowledge/*
│       │   └── feedback.py          # POST /api/feedback
│       ├── db/
│       │   ├── client.py            # AsyncSurreal wrapper
│       │   ├── schema.surql          # Full schema definition
│       │   ├── queries.py           # Named query helpers
│       │   └── seed.py              # Knowledge graph seeder
│       ├── middleware/
│       │   ├── auth.py              # API key authentication
│       │   ├── validation.py        # Input validation + PII redaction
│       │   ├── rate_limit.py        # Per-user rate limiting
│       │   └── audit.py             # LLM call audit logging
│       ├── prompts/
│       │   └── manager.py           # Prompt template management
│       └── tracing/
│           ├── opik_setup.py        # Opik initialisation
│           ├── opik_evaluator.py    # 3-stage quality gates
│           ├── langsmith_setup.py   # LangSmith tracing config
│           └── cost_tracker.py      # Token + USD cost tracking
│
├── frontend/
│   ├── package.json
│   ├── next.config.ts
│   ├── vitest.config.ts
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   └── page.tsx             # Split view: Chat | Dashboard
│       ├── components/
│       │   ├── Chat.tsx             # AI SDK useChat integration
│       │   ├── RiskDashboard.tsx    # Risk score + recommendations
│       │   ├── KnowledgeGraph.tsx   # Interactive graph visualisation
│       │   ├── ScenarioPanel.tsx    # Scenario simulator
│       │   └── GraphViz.tsx         # Graph rendering utility
│       └── lib/
│           └── api.ts               # Backend API client
│
└── tests are co-located in backend/tests/ and frontend/src/__tests__/
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for SurrealDB)

### Local Development

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your API keys (see Environment Variables below)

# 2. Start SurrealDB
docker compose up surrealdb -d

# 3. Start the backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8080 --reload

# 4. Start the frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Docker Compose (full stack)

```bash
cp .env.example .env
# Edit .env with your API keys
docker compose up
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8080 |
| SurrealDB | http://localhost:8000 |

---

## Environment Variables

Create a `.env` file from `.env.example`. The following variables are used:

### Backend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for Claude |
| `SURREAL_URL` | No | `ws://localhost:8000/rpc` | SurrealDB WebSocket endpoint |
| `SURREAL_USER` | No | `root` | SurrealDB username |
| `SURREAL_PASS` | No | `root` | SurrealDB password |
| `SURREAL_NS` | No | `vigil` | SurrealDB namespace |
| `SURREAL_DB` | No | `vigil` | SurrealDB database |
| `OPIK_API_KEY` | No | — | Comet Opik key (enables quality gates) |
| `OPIK_WORKSPACE` | No | — | Opik workspace name |
| `OPIK_PROJECT_NAME` | No | `vigil` | Opik project name |
| `LANGCHAIN_TRACING_V2` | No | `true` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | `vigil` | LangSmith project name |

### Frontend

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key for AI SDK chat |
| `BACKEND_URL` | No | `http://localhost:8080` | Backend API URL (server-side) |
| `BACKEND_API_KEY` | No | `pg-demo-key-2025` | API key for backend calls |
| `NEXT_PUBLIC_BACKEND_URL` | No | `http://localhost:8080` | Backend API URL (client-side) |

---

## API Endpoints

All routes are prefixed with `/api`.

### Assessment

```
POST /api/assess
```

Run a full 6-stage risk assessment. Request body:

```json
{
  "description": "Customer support chatbot with access to order database...",
  "jurisdictions": ["UK", "EU"],
  "sector": "financial_services"
}
```

Returns the complete assessment: deployment profile, legal/technical/mitigation analyses, risk price, and session ID.

### Scenarios

```
POST /api/scenario
```

Run a risk scenario against an existing assessment:

```json
{
  "session_id": "abc-123",
  "scenario_type": "data_breach"
}
```

Valid scenario types: `hallucination`, `contract_formation`, `data_breach`, `scope_creep`, `adversarial`, `regulatory_breach`.

### Knowledge Graph

```
GET /api/knowledge/full          # Complete graph for visualisation
GET /api/knowledge/stats         # Summary counts
GET /api/knowledge/doctrines     # Query by jurisdiction (?jurisdiction=UK)
GET /api/knowledge/regulations   # Query by jurisdiction
GET /api/knowledge/risk-factors  # Optional ?category= filter
GET /api/knowledge/mitigations/{risk_factor}
GET /api/knowledge/doctrine/{name}/relationships
GET /api/knowledge/audit/{session_id}
```

### Feedback

```
POST /api/feedback               # Submit feedback (thumbs_up/thumbs_down/override)
GET  /api/feedback/{session_id}  # Get feedback for a session
```

### Health

```
GET /health                      # Returns {"status": "ok"}
```

---

## Testing

### Backend (pytest)

```bash
cd backend
pytest                          # All tests
pytest -m "not integration"     # Unit tests only
pytest -m integration           # Integration tests (requires running SurrealDB)
```

### Frontend (vitest)

```bash
cd frontend
npm test                        # Run all tests
npm run test:ui                 # Interactive UI mode
```

---

## Observability

### Opik Quality Gates

Each LLM stage is validated before results propagate. Three evaluators run automatically:

- **Legal analysis** — Checks cited doctrines exist in the knowledge graph, exposure scores are bounded, jurisdiction-specific rules are applied (e.g. GDPR for PII access, apparent authority for communication tools)
- **Technical analysis** — Verifies all risk factors are scored, scores are bounded, amplification effects reference real factors
- **Pricing** — Validates risk score bounds, premium band consistency, scenario/recommendation completeness

Quality threshold: **0.5** — assessments below this are flagged.

### LangSmith Tracing

When `LANGCHAIN_API_KEY` is set, all LangGraph invocations are traced in LangSmith with full input/output visibility per node.

### Cost Tracking

Every LLM call logs token usage and USD cost (based on Claude Sonnet pricing) to the `audit_log` table. Aggregated views available:

- Per-session cost via `get_session_cost()`
- Daily cost summary via `get_daily_cost()`
- Average cost per assessment via `get_average_cost_per_run()`

---

## License

Private — all rights reserved.
