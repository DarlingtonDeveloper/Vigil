from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.client import db
from app.db.seed import seed_knowledge_graph
from app.prompts.manager import seed_default_prompts
from app.tracing.langsmith_setup import verify_langsmith_config
from app.tracing.opik_setup import init_opik


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await db.connect()
        schema_path = Path(__file__).parent / "db" / "schema.surql"
        await db.execute_schema(str(schema_path))
        await seed_knowledge_graph()
        await seed_default_prompts()
        print("SurrealDB: CONNECTED and seeded")
    except Exception as e:
        print(f"SurrealDB: STARTUP ERROR — {e}")

    langsmith_status = verify_langsmith_config()
    print(f"LangSmith: {'READY' if langsmith_status['ready'] else 'NOT CONFIGURED'}")

    try:
        init_opik()
        print("Opik: INITIALIZED")
    except Exception as e:
        print(f"Opik: INIT ERROR — {e}")

    yield

    # Shutdown
    await db.disconnect()


app = FastAPI(
    title="Vigil",
    description="Agentic risk pricing engine for AI agent deployments",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from app.routes import assess, scenario, knowledge, feedback

app.include_router(assess.router)
app.include_router(scenario.router)
app.include_router(knowledge.router)
app.include_router(feedback.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
