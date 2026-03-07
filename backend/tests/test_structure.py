import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def test_backend_app_packages():
    packages = [
        "backend/app",
        "backend/app/agents",
        "backend/app/graph",
        "backend/app/db",
        "backend/app/middleware",
        "backend/app/tracing",
        "backend/app/prompts",
        "backend/app/routes",
    ]
    for pkg in packages:
        init = ROOT / pkg / "__init__.py"
        assert init.exists(), f"Missing {init}"


def test_backend_stub_files():
    stubs = [
        "backend/app/agents/intake.py",
        "backend/app/agents/legal.py",
        "backend/app/agents/technical.py",
        "backend/app/agents/mitigation.py",
        "backend/app/agents/pricing.py",
        "backend/app/graph/workflow.py",
        "backend/app/graph/state.py",
        "backend/app/db/client.py",
        "backend/app/db/queries.py",
        "backend/app/db/schema.surql",
        "backend/app/db/seed.py",
        "backend/app/middleware/auth.py",
        "backend/app/middleware/rate_limit.py",
        "backend/app/middleware/validation.py",
        "backend/app/middleware/audit.py",
        "backend/app/tracing/opik_setup.py",
        "backend/app/tracing/opik_evaluator.py",
        "backend/app/tracing/langsmith_setup.py",
        "backend/app/tracing/cost_tracker.py",
        "backend/app/prompts/manager.py",
        "backend/app/routes/assess.py",
        "backend/app/routes/scenario.py",
        "backend/app/routes/knowledge.py",
        "backend/app/routes/feedback.py",
    ]
    for stub in stubs:
        path = ROOT / stub
        assert path.exists(), f"Missing {path}"


def test_prompt_templates():
    templates = ["intake.md", "legal.md", "technical.md", "mitigation.md", "pricing.md"]
    for name in templates:
        path = ROOT / "backend" / "prompts" / "v1" / name
        assert path.exists(), f"Missing {path}"


def test_frontend_files():
    files = [
        "frontend/package.json",
        "frontend/next.config.ts",
        "frontend/tsconfig.json",
        "frontend/tailwind.config.ts",
        "frontend/src/app/layout.tsx",
        "frontend/src/app/page.tsx",
        "frontend/src/app/api/chat/route.ts",
        "frontend/src/components/Chat.tsx",
        "frontend/src/components/KnowledgeGraph.tsx",
        "frontend/src/components/RiskDashboard.tsx",
        "frontend/src/lib/api.ts",
    ]
    for f in files:
        path = ROOT / f
        assert path.exists(), f"Missing {path}"


def test_infrastructure_files():
    files = [
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
        "backend/Dockerfile",
        "backend/requirements.txt",
        "backend/.env.example",
    ]
    for f in files:
        path = ROOT / f
        assert path.exists(), f"Missing {path}"
