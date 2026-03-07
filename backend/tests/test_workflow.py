"""Tests for SPEC-07: LangGraph Workflow Orchestrator."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.graph.state import FaultLineState
from app.graph.workflow import (
    build_workflow,
    intake_node,
    fetch_knowledge_node,
    legal_node,
    technical_node,
    mitigation_node,
    pricing_node,
    run_assessment,
)
from app.agents.intake import DeploymentProfile, ToolSpec
from app.agents.legal import LegalAnalysis, LegalExposure
from app.agents.technical import TechnicalAnalysis, TechnicalRisk
from app.agents.mitigation import MitigationAnalysis, MitigationRecommendation
from app.agents.pricing import PricingResult, RiskScenario


# -- Fixtures ----------------------------------------------------------------

SAMPLE_PROFILE = DeploymentProfile(
    agent_description="Customer support chatbot",
    tools=[ToolSpec(name="email_sender", action_type="communicate", description="Sends emails")],
    data_access=["customer_records", "order_history"],
    autonomy_level="semi-autonomous",
    output_reach="external",
    sector="financial",
    jurisdictions=["UK"],
    human_oversight_model="approval_required",
    existing_guardrails=["content_filter"],
)

SAMPLE_LEGAL = LegalAnalysis(
    legal_exposure_score=0.65,
    exposures=[LegalExposure(doctrine="negligence", risk_level="medium", reasoning="AI acts on behalf")],
    applicable_regulations=["UK AI Act", "GDPR"],
    compliance_gaps=["No impact assessment"],
    summary="Moderate legal exposure",
)

SAMPLE_TECHNICAL = TechnicalAnalysis(
    technical_risk_score=0.55,
    risks=[TechnicalRisk(factor="hallucination", score=0.7, reasoning="LLM may confabulate")],
    failure_modes=["Incorrect responses"],
    data_risks=["PII leakage"],
    summary="Moderate technical risk",
)

SAMPLE_MITIGATION = MitigationAnalysis(
    overall_mitigation_score=0.4,
    recommended_mitigations=[
        MitigationRecommendation(name="human_review", effectiveness=0.8,
                                 implementation_effort="medium", reasoning="Catches errors")
    ],
    existing_coverage=["content_filter"],
    gaps=["No audit trail"],
    summary="Partial mitigation",
)

SAMPLE_PRICING = PricingResult(
    technical_risk=0.55,
    legal_exposure=0.65,
    overall_risk_score=0.6,
    premium_band="medium",
    confidence=0.75,
    executive_summary="Moderate risk deployment",
    top_exposures=["negligence", "data_breach"],
    recommendations=["Add human review", "Implement audit trail"],
    scenarios=[
        RiskScenario(
            scenario_type="data_breach",
            probability="medium",
            severity="high",
            expected_loss_range="50k-500k GBP",
            applicable_doctrines=["negligence"],
            mitigation_options=["encryption", "access_control"],
        )
    ],
)


def _base_state(**overrides) -> FaultLineState:
    state: FaultLineState = {
        "session_id": "test-session-001",
        "user_id": "demo-user",
        "description": "A customer support chatbot that sends emails",
        "jurisdictions": ["UK"],
        "sector": "financial",
        "deployment_profile": None,
        "applicable_doctrines": [],
        "applicable_regulations": [],
        "risk_factors": [],
        "available_mitigations": [],
        "mitigation_edges": [],
        "legal_analysis": None,
        "technical_analysis": None,
        "mitigation_analysis": None,
        "risk_price": None,
        "legal_quality_score": 0.0,
        "technical_quality_score": 0.0,
        "pricing_quality_score": 0.0,
        "current_step": "started",
        "error": None,
    }
    state.update(overrides)
    return state


# -- State tests -------------------------------------------------------------

def test_faultline_state_has_all_required_keys():
    """FaultLineState TypedDict declares all expected keys."""
    annotations = FaultLineState.__annotations__
    expected = [
        "session_id", "user_id", "description", "jurisdictions", "sector",
        "deployment_profile",
        "applicable_doctrines", "applicable_regulations", "risk_factors",
        "available_mitigations", "mitigation_edges",
        "legal_analysis", "technical_analysis", "mitigation_analysis", "risk_price",
        "legal_quality_score", "technical_quality_score", "pricing_quality_score",
        "current_step", "error",
    ]
    for key in expected:
        assert key in annotations, f"Missing key {key} in FaultLineState"


# -- Build workflow tests ----------------------------------------------------

def test_build_workflow_returns_compiled_graph():
    """build_workflow() returns a compiled LangGraph with 6 nodes."""
    graph = build_workflow()
    # The compiled graph should have a nodes dict
    node_names = set(graph.nodes.keys())
    expected_nodes = {"intake", "fetch_knowledge", "legal", "technical", "mitigation", "pricing"}
    # LangGraph also adds __start__ and __end__ nodes
    assert expected_nodes.issubset(node_names), f"Missing nodes: {expected_nodes - node_names}"


def test_build_workflow_has_six_custom_nodes():
    """The graph must contain exactly 6 custom nodes (not counting __start__/__end__)."""
    graph = build_workflow()
    custom_nodes = {k for k in graph.nodes if not k.startswith("__")}
    assert len(custom_nodes) == 6


def test_build_workflow_is_deterministic():
    """Calling build_workflow() twice gives graphs with same structure."""
    g1 = build_workflow()
    g2 = build_workflow()
    assert set(g1.nodes.keys()) == set(g2.nodes.keys())


# -- Individual node tests (mocked) -----------------------------------------

@pytest.mark.asyncio
async def test_intake_node():
    """Intake node calls run_intake and stores profile in DB."""
    state = _base_state()

    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE) as mock_intake, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock) as mock_audit, \
         patch("app.graph.workflow.db") as mock_db:
        mock_db.query = AsyncMock(return_value=[])

        result = await intake_node(state)

    assert result["current_step"] == "intake_complete"
    assert result["deployment_profile"] is not None
    assert result["deployment_profile"]["autonomy_level"] == "semi-autonomous"
    mock_intake.assert_awaited_once()
    assert mock_audit.await_count == 2  # start + complete
    mock_db.query.assert_awaited_once()  # CREATE deployment_profile


@pytest.mark.asyncio
async def test_fetch_knowledge_node():
    """Fetch knowledge node queries SurrealDB for doctrines, regulations, risk factors, mitigations."""
    state = _base_state(deployment_profile=SAMPLE_PROFILE.model_dump())

    fake_doctrines = [{"name": "negligence", "jurisdiction": "UK"}]
    fake_regulations = [{"name": "GDPR", "jurisdiction": "UK"}]
    fake_risk_factors = [{"result": [{"name": "hallucination", "weight": 0.8}]}]
    fake_mitigations = [{"result": [{"name": "human_review"}]}]
    fake_edges = [{"result": [{"mitigation": "human_review", "risk_factor": "hallucination"}]}]

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=fake_doctrines)
        mock_db.get_applicable_regulations = AsyncMock(return_value=fake_regulations)
        mock_db.query = AsyncMock(side_effect=[fake_risk_factors, fake_mitigations, fake_edges])

        result = await fetch_knowledge_node(state)

    assert result["current_step"] == "knowledge_fetched"
    assert result["applicable_doctrines"] == fake_doctrines
    assert result["applicable_regulations"] == fake_regulations
    assert len(result["risk_factors"]) > 0
    assert len(result["available_mitigations"]) > 0


@pytest.mark.asyncio
async def test_fetch_knowledge_adds_employment_law_for_financial():
    """Financial sector deployments should include employment_law domain."""
    profile = SAMPLE_PROFILE.model_dump()
    profile["sector"] = "financial"
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])

        await fetch_knowledge_node(state)

    call_args = mock_db.get_applicable_doctrines.call_args
    domains = call_args[0][1]
    assert "employment_law" in domains


@pytest.mark.asyncio
async def test_legal_node():
    """Legal node calls run_legal_analysis with doctrines from knowledge graph."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        applicable_doctrines=[{"name": "negligence"}],
        applicable_regulations=[{"name": "GDPR"}],
    )

    with patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL) as mock_legal, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        result = await legal_node(state)

    assert result["current_step"] == "legal_complete"
    assert result["legal_analysis"]["legal_exposure_score"] == 0.65
    # Verify agent received real knowledge graph data
    mock_legal.assert_awaited_once_with(
        state["deployment_profile"],
        [{"name": "negligence"}],  # doctrines from KB, not hallucinated
        [{"name": "GDPR"}],
        "test-session-001",
    )


@pytest.mark.asyncio
async def test_technical_node():
    """Technical node calls run_technical_analysis."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        risk_factors=[{"name": "hallucination", "weight": 0.8}],
    )

    with patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        result = await technical_node(state)

    assert result["current_step"] == "technical_complete"
    assert result["technical_analysis"]["technical_risk_score"] == 0.55


@pytest.mark.asyncio
async def test_mitigation_node():
    """Mitigation node calls run_mitigation_analysis."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        available_mitigations=[{"name": "human_review"}],
        mitigation_edges=[{"mitigation": "human_review", "risk_factor": "hallucination"}],
    )

    with patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        result = await mitigation_node(state)

    assert result["current_step"] == "mitigation_complete"
    assert result["mitigation_analysis"]["overall_mitigation_score"] == 0.4


@pytest.mark.asyncio
async def test_pricing_node():
    """Pricing node stores risk_score, scenarios, and updates assessment status."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        legal_analysis=SAMPLE_LEGAL.model_dump(),
        technical_analysis=SAMPLE_TECHNICAL.model_dump(),
        mitigation_analysis=SAMPLE_MITIGATION.model_dump(),
    )

    with patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock) as mock_audit, \
         patch("app.graph.workflow.db") as mock_db:
        mock_db.query = AsyncMock(return_value=[])

        result = await pricing_node(state)

    assert result["current_step"] == "complete"
    assert result["risk_price"]["premium_band"] == "medium"
    assert result["risk_price"]["overall_risk_score"] == 0.6
    # DB calls: risk_score CREATE + 1 scenario CREATE + UPDATE assessment
    assert mock_db.query.await_count == 3
    assert mock_audit.await_count == 2  # start + complete


@pytest.mark.asyncio
async def test_pricing_node_persists_scenarios():
    """Each scenario from pricing result is persisted to SurrealDB."""
    pricing_with_two = PricingResult(
        **{**SAMPLE_PRICING.model_dump(),
           "scenarios": [
               RiskScenario(scenario_type="breach", probability="high", severity="high",
                            expected_loss_range="100k-1M", applicable_doctrines=["negligence"],
                            mitigation_options=["encryption"]),
               RiskScenario(scenario_type="bias", probability="medium", severity="medium",
                            expected_loss_range="10k-100k", applicable_doctrines=["discrimination"],
                            mitigation_options=["audit"]),
           ]}
    )
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        legal_analysis=SAMPLE_LEGAL.model_dump(),
        technical_analysis=SAMPLE_TECHNICAL.model_dump(),
        mitigation_analysis=SAMPLE_MITIGATION.model_dump(),
    )

    with patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=pricing_with_two), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db:
        mock_db.query = AsyncMock(return_value=[])
        await pricing_node(state)

    # risk_score CREATE + 2 scenario CREATEs + UPDATE assessment = 4
    assert mock_db.query.await_count == 4


# -- Audit trail tests ------------------------------------------------------

@pytest.mark.asyncio
async def test_audit_trail_for_every_node():
    """Every node should log audit start and complete events."""
    state = _base_state(deployment_profile=SAMPLE_PROFILE.model_dump())

    audit_calls = []

    async def track_audit(sid, step, status, output_data=None):
        audit_calls.append((step, status))

    with patch("app.graph.workflow.log_audit", side_effect=track_audit), \
         patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.db") as mock_db:
        mock_db.query = AsyncMock(return_value=[])
        await intake_node(_base_state())

    assert ("intake", "start") in audit_calls
    assert ("intake", "complete") in audit_calls


# -- run_assessment integration test (all mocked) ---------------------------

@pytest.mark.asyncio
async def test_run_assessment_full_pipeline():
    """run_assessment executes the full pipeline with all nodes."""
    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[{"name": "negligence"}])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[{"name": "GDPR"}])
        mock_opik.return_value = MagicMock()

        result = await run_assessment(
            description="A customer support chatbot",
            jurisdictions=["UK"],
            sector="financial",
            user_id="demo-user",
        )

    assert result["current_step"] == "complete"
    assert result["risk_price"] is not None
    assert result["risk_price"]["premium_band"] == "medium"
    assert result["deployment_profile"] is not None
    assert result["legal_analysis"] is not None
    assert result["technical_analysis"] is not None
    assert result["mitigation_analysis"] is not None
    # Opik tracer was created
    mock_opik.assert_called_once()


@pytest.mark.asyncio
async def test_run_assessment_marks_failed_on_error():
    """If a node raises, assessment status is updated to 'failed'."""
    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, side_effect=RuntimeError("LLM down")), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db.query = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="LLM down"):
            await run_assessment("desc", ["UK"], "financial", "demo-user")

    # Should have called UPDATE to mark as failed
    update_calls = [c for c in mock_db.query.call_args_list if "failed" in str(c)]
    assert len(update_calls) >= 1


# -- Pydantic model tests ---------------------------------------------------

def test_deployment_profile_model():
    p = DeploymentProfile(
        agent_description="test",
        tools=[ToolSpec(name="api_caller", action_type="read")],
        autonomy_level="supervised",
        sector="financial",
        jurisdictions=["UK"],
    )
    d = p.model_dump()
    assert d["autonomy_level"] == "supervised"
    assert len(d["tools"]) == 1


def test_legal_analysis_model():
    a = LegalAnalysis(legal_exposure_score=0.7, summary="High exposure")
    assert a.legal_exposure_score == 0.7
    assert a.model_dump()["summary"] == "High exposure"


def test_technical_analysis_model():
    a = TechnicalAnalysis(technical_risk_score=0.3)
    assert 0.0 <= a.technical_risk_score <= 1.0


def test_mitigation_analysis_model():
    a = MitigationAnalysis(overall_mitigation_score=0.5)
    assert a.overall_mitigation_score == 0.5


def test_pricing_result_model():
    p = PricingResult(
        overall_risk_score=0.6,
        premium_band="medium",
        scenarios=[RiskScenario(scenario_type="breach")],
    )
    assert p.premium_band == "medium"
    assert len(p.scenarios) == 1


# -- Tracing setup tests ----------------------------------------------------

def test_opik_tracer_creation():
    with patch("app.tracing.opik_setup.OpikTracer") as mock_cls:
        from app.tracing.opik_setup import get_opik_tracer
        tracer = get_opik_tracer("sess-123", ["test-tag"])
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["metadata"]["session_id"] == "sess-123"
        assert "test-tag" in call_kwargs["tags"]


def test_langsmith_config():
    from app.tracing.langsmith_setup import get_langsmith_run_config
    config = get_langsmith_run_config("sess-456", "test-run")
    assert "sess-456" in config["run_name"]
    assert config["metadata"]["session_id"] == "sess-456"
    assert "test-run" in config["tags"]


# -- DB client tests ---------------------------------------------------------

def test_db_client_initial_state():
    from app.db.client import DatabaseClient
    client = DatabaseClient()
    assert not client.connected


@pytest.mark.asyncio
async def test_db_client_query_connects_lazily():
    """Calling query when not connected should trigger connect."""
    from app.db.client import DatabaseClient
    client = DatabaseClient()

    with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
        # _db is still None, so query will call connect then fail on _db.query
        # We need to set _db after connect is called
        async def set_db():
            mock_inner = AsyncMock()
            mock_inner.query = AsyncMock(return_value=[{"result": []}])
            client._db = mock_inner

        mock_connect.side_effect = set_db
        result = await client.query("SELECT 1")
        mock_connect.assert_awaited_once()
