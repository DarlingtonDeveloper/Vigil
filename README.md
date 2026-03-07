# FaultLine

Agentic risk pricing engine for AI agent deployments.

## Quick Start

```bash
# Copy environment files
cp .env.example .env

# Start SurrealDB
docker compose up surrealdb

# Start backend (in another terminal)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8080

# Start frontend (in another terminal)
cd frontend
npm install
npm run dev
```

## Docker Compose (full stack)

```bash
docker compose up
```

- SurrealDB: http://localhost:8000
- Backend API: http://localhost:8080
- Frontend: http://localhost:3000
