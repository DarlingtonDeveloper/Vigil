from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.tracing.langsmith_setup import verify_langsmith_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    langsmith_status = verify_langsmith_config()
    print(f"LangSmith: {'READY' if langsmith_status['ready'] else 'NOT CONFIGURED'}")
    yield


app = FastAPI(
    title="PortfolioGraph",
    description="Agentic portfolio contagion analyzer",
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


@app.get("/health")
async def health():
    return {"status": "ok"}

# Routes will be added in SPEC-08
