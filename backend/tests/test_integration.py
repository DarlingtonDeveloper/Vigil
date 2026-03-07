"""
Integration tests — run against a live SurrealDB instance.

These tests require SurrealDB running at ws://localhost:8000.
Skip with: pytest -m "not integration"

Start SurrealDB:
    surreal start --user root --pass root memory
"""
import pytest

# Mark all tests in this module as integration
pytestmark = pytest.mark.integration


@pytest.fixture
async def live_client():
    """Connect to a real SurrealDB, use an isolated test namespace/db, and clean up."""
    from app.db.client import SurrealClient
    c = SurrealClient()
    c._namespace = "test_faultline"
    c._database = "test_faultline"
    await c.connect()
    await c.execute_schema()
    yield c
    await c.query("REMOVE DATABASE test_faultline")
    await c.disconnect()


class TestSchemaExecution:
    """Verify schema.surql runs without errors."""

    async def test_schema_creates_tables(self, live_client):
        result = await live_client.query("INFO FOR DB")
        assert result is not None

    async def test_schema_idempotent(self, live_client):
        await live_client.execute_schema()


class TestSeedIntegration:
    """Verify seeding works against a real database."""

    async def test_full_seed(self, live_client):
        import app.db.seed as seed_module
        original_db = seed_module.db
        seed_module.db = live_client
        try:
            await seed_module.seed_knowledge_graph()
        finally:
            seed_module.db = original_db

        domains = await live_client.select("legal_domain")
        assert len(domains) == 6
        doctrines = await live_client.select("doctrine")
        assert len(doctrines) == 8
        regulations = await live_client.select("regulation")
        assert len(regulations) == 5
        risk_factors = await live_client.select("risk_factor")
        assert len(risk_factors) == 10
        mitigations = await live_client.select("mitigation")
        assert len(mitigations) == 15

    async def test_seed_idempotent(self, live_client):
        import app.db.seed as seed_module
        original_db = seed_module.db
        seed_module.db = live_client
        try:
            await seed_module.seed_knowledge_graph()
            await seed_module.seed_knowledge_graph()
        finally:
            seed_module.db = original_db

        domains = await live_client.select("legal_domain")
        assert len(domains) == 6


class TestHelperQueriesIntegration:
    """Verify helper queries work against seeded data."""

    @pytest.fixture(autouse=True)
    async def seed_data(self, live_client):
        import app.db.seed as seed_module
        original_db = seed_module.db
        seed_module.db = live_client
        try:
            await seed_module.seed_knowledge_graph()
        finally:
            seed_module.db = original_db

    async def test_get_applicable_doctrines_uk_contract(self, live_client):
        result = await live_client.get_applicable_doctrines(["UK"], ["contract_law"])
        assert len(result) > 0

    async def test_get_applicable_regulations_eu(self, live_client):
        result = await live_client.get_applicable_regulations(["EU"])
        assert len(result) > 0

    async def test_get_risk_factors_by_category_technical(self, live_client):
        result = await live_client.get_risk_factors_by_category("technical")
        assert len(result) > 0

    async def test_get_mitigations_for_hallucination_risk(self, live_client):
        result = await live_client.get_mitigations_for_risk("hallucination_risk")
        assert len(result) > 0

    async def test_get_knowledge_graph_full(self, live_client):
        graph = await live_client.get_knowledge_graph_full()
        assert "doctrines" in graph
        assert "regulations" in graph
        assert "risk_factors" in graph
        assert "mitigations" in graph
        assert "doctrine_edges" in graph
        assert "mitigation_edges" in graph
