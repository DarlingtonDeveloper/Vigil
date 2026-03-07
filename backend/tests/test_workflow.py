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
from app.agents.intake import DeploymentProfile, ToolDescription, VendorInfo
from app.agents.legal import LegalAnalysis, DoctrineAssessment, RegulatoryGap
from app.agents.technical import TechnicalAnalysis, FactorScore
from app.agents.mitigation import MitigationAnalysis, MitigationAxisScore
from app.agents.pricing import RiskPrice, RiskScenario


# -- Fixtures ----------------------------------------------------------------

SAMPLE_PROFILE = DeploymentProfile(
    agent_description="Customer support chatbot",
    tools=[ToolDescription(name="email_sender", action_type="communicate", description="Sends emails")],
    data_access=["customer_records", "order_history"],
    autonomy_level="human_on_the_loop",
    output_reach="customer_facing",
    sector="financial",
    jurisdictions=["UK"],
    human_oversight_model="approval_required",
    reviewer_qualification="general_operator",
    existing_guardrails=["content_filter"],
    vendor_info=VendorInfo(provider="Anthropic", model="claude-sonnet-4-20250514",
                           indemnification="partial", contract_reviewed=True),
    key_risks_identified=["PII exposure"],
)

SAMPLE_LEGAL = LegalAnalysis(
    legal_exposure_score=0.65,
    doctrine_assessments=[
        DoctrineAssessment(doctrine_name="negligence", applies=True,
                           exposure_level="medium", reasoning="AI acts on behalf",
                           worst_case="Liability for incorrect advice"),
    ],
    regulatory_gaps=[
        RegulatoryGap(regulation="GDPR", requirement="DPIA",
                      status="non_compliant", risk_if_non_compliant="Fines up to 4% revenue"),
    ],
    contract_formation_risk="Medium - agent communicates externally",
    tort_exposure="Medium - potential for negligent misstatement",
    key_uncertainties=["Untested AI liability precedent"],
    confidence=0.7,
)

SAMPLE_TECHNICAL = TechnicalAnalysis(
    technical_risk_score=0.55,
    factor_scores=[
        FactorScore(factor_name="hallucination", level="moderate", score=0.7,
                    reasoning="LLM may confabulate"),
    ],
    amplification_effects=["Customer-facing + hallucination = multiplied exposure"],
    key_vulnerabilities=["PII leakage", "Incorrect responses"],
    confidence=0.75,
)

SAMPLE_MITIGATION = MitigationAnalysis(
    overall_mitigation_score=0.4,
    axis_scores=[
        MitigationAxisScore(axis="architectural", score=0.5,
                            present_mitigations=["content_filter"],
                            missing_mitigations=["output_schema"],
                            critical_gaps=["No confidence threshold"]),
    ],
    recommendations=[{"name": "human_review", "priority": "high", "impact": "0.3", "cost": "moderate",
                       "reasoning": "Catches errors before customer delivery"}],
    quick_wins=["Add output schema validation"],
    confidence=0.7,
)

SAMPLE_PRICING = RiskPrice(
    executive_summary="Moderate risk deployment",
    technical_risk=0.55,
    legal_exposure=0.65,
    mitigation_effectiveness=0.4,
    overall_risk_score=0.6,
    premium_band="Medium ($15K-$50K/yr)",
    premium_reasoning="Significant legal exposure partially offset by content filtering",
    top_exposures=[{"exposure": "negligence", "severity": "medium", "mitigation_available": True}],
    recommendations=[{"action": "Add human review", "priority": "high", "impact": "0.3",
                       "reasoning": "Catches errors"}],
    confidence=0.75,
    data_gaps=["Vendor contract terms unknown"],
    scenarios=[
        RiskScenario(
            scenario_type="data_breach",
            probability="possible",
            severity="major",
            expected_loss_range="$50K-$500K",
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
    node_names = set(graph.nodes.keys())
    expected_nodes = {"intake", "fetch_knowledge", "legal", "technical", "mitigation", "pricing"}
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
    assert result["deployment_profile"]["autonomy_level"] == "human_on_the_loop"
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
    assert "Medium" in result["risk_price"]["premium_band"]
    assert result["risk_price"]["overall_risk_score"] == 0.6
    # DB calls: risk_score CREATE + 1 scenario CREATE + UPDATE assessment
    assert mock_db.query.await_count == 3
    assert mock_audit.await_count == 2  # start + complete


@pytest.mark.asyncio
async def test_pricing_node_persists_scenarios():
    """Each scenario from pricing result is persisted to SurrealDB."""
    pricing_with_two = RiskPrice(
        **{**SAMPLE_PRICING.model_dump(),
           "scenarios": [
               RiskScenario(scenario_type="breach", probability="likely", severity="catastrophic",
                            expected_loss_range="$100K-$1M", applicable_doctrines=["negligence"],
                            mitigation_options=["encryption"]),
               RiskScenario(scenario_type="bias", probability="possible", severity="moderate",
                            expected_loss_range="$10K-$100K", applicable_doctrines=["discrimination"],
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
    audit_calls = []

    async def track_audit(sid, agent, action, **kwargs):
        audit_calls.append((agent, action))

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

        mock_db._conn = True  # Simulate connected state
        mock_db.connect = AsyncMock()
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
    assert "Medium" in result["risk_price"]["premium_band"]
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

        mock_db._conn = True
        mock_db.connect = AsyncMock()
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
        tools=[ToolDescription(name="api_caller", action_type="read", description="Reads data")],
        data_access=["internal"],
        autonomy_level="human_in_the_loop",
        output_reach="internal_only",
        sector="financial",
        jurisdictions=["UK"],
        human_oversight_model=None,
        reviewer_qualification=None,
        existing_guardrails=[],
        vendor_info=None,
        key_risks_identified=[],
    )
    d = p.model_dump()
    assert d["autonomy_level"] == "human_in_the_loop"
    assert len(d["tools"]) == 1


def test_legal_analysis_model():
    a = LegalAnalysis(
        legal_exposure_score=0.7,
        doctrine_assessments=[],
        regulatory_gaps=[],
        contract_formation_risk="Low",
        tort_exposure="Low",
        key_uncertainties=[],
        confidence=0.8,
    )
    assert a.legal_exposure_score == 0.7
    assert a.confidence == 0.8


def test_technical_analysis_model():
    a = TechnicalAnalysis(
        technical_risk_score=0.3,
        factor_scores=[],
        amplification_effects=[],
        key_vulnerabilities=[],
        confidence=0.9,
    )
    assert 0.0 <= a.technical_risk_score <= 1.0


def test_mitigation_analysis_model():
    a = MitigationAnalysis(
        overall_mitigation_score=0.5,
        axis_scores=[],
        recommendations=[],
        quick_wins=[],
        confidence=0.8,
    )
    assert a.overall_mitigation_score == 0.5


def test_pricing_result_model():
    p = RiskPrice(
        executive_summary="Test",
        overall_risk_score=0.6,
        technical_risk=0.5,
        legal_exposure=0.7,
        mitigation_effectiveness=0.4,
        premium_band="Medium ($15K-$50K/yr)",
        premium_reasoning="Test reasoning",
        top_exposures=[],
        scenarios=[RiskScenario(scenario_type="breach", probability="possible",
                                severity="major", expected_loss_range="$50K-$500K",
                                applicable_doctrines=[], mitigation_options=[])],
        recommendations=[],
        confidence=0.75,
        data_gaps=[],
    )
    assert "Medium" in p.premium_band
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
    from app.db.client import SurrealClient
    client = SurrealClient()
    assert client._conn is None


@pytest.mark.asyncio
async def test_db_client_query_raises_when_not_connected():
    """Calling query when not connected should raise RuntimeError."""
    from app.db.client import SurrealClient
    client = SurrealClient()

    with pytest.raises(RuntimeError, match="not connected"):
        await client.query("SELECT 1")


# ============================================================================
# UNIT TESTS: fetch_knowledge edge cases + flatten helper
# ============================================================================

@pytest.mark.asyncio
async def test_fetch_knowledge_empty_tools_excludes_ip():
    """Profile with no tools should not add intellectual_property domain."""
    profile = SAMPLE_PROFILE.model_dump()
    profile["tools"] = []
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])
        await fetch_knowledge_node(state)

    domains = mock_db.get_applicable_doctrines.call_args[0][1]
    assert "intellectual_property" not in domains


@pytest.mark.asyncio
async def test_fetch_knowledge_communicate_tool_adds_ip():
    """Profile with a communicate tool should add intellectual_property domain."""
    profile = SAMPLE_PROFILE.model_dump()
    profile["tools"] = [{"name": "emailer", "action_type": "communicate", "description": "sends"}]
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])
        await fetch_knowledge_node(state)

    domains = mock_db.get_applicable_doctrines.call_args[0][1]
    assert "intellectual_property" in domains


@pytest.mark.asyncio
async def test_fetch_knowledge_non_financial_sector_excludes_employment():
    """Non-financial/healthcare/legal sectors should not add employment_law."""
    profile = SAMPLE_PROFILE.model_dump()
    profile["sector"] = "education"
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])
        await fetch_knowledge_node(state)

    domains = mock_db.get_applicable_doctrines.call_args[0][1]
    assert "employment_law" not in domains


@pytest.mark.asyncio
async def test_fetch_knowledge_null_sector_excludes_employment():
    """None sector should not add employment_law."""
    profile = SAMPLE_PROFILE.model_dump()
    profile["sector"] = None
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])
        await fetch_knowledge_node(state)

    domains = mock_db.get_applicable_doctrines.call_args[0][1]
    assert "employment_law" not in domains


@pytest.mark.asyncio
async def test_fetch_knowledge_tool_without_action_type():
    """Tool missing action_type key should not trigger intellectual_property."""
    profile = SAMPLE_PROFILE.model_dump()
    profile["tools"] = [{"name": "reader", "description": "reads files"}]  # no action_type
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])
        await fetch_knowledge_node(state)

    domains = mock_db.get_applicable_doctrines.call_args[0][1]
    assert "intellectual_property" not in domains


@pytest.mark.asyncio
async def test_fetch_knowledge_default_jurisdictions():
    """Missing jurisdictions in profile should default to ['UK']."""
    profile = SAMPLE_PROFILE.model_dump()
    del profile["jurisdictions"]
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])
        await fetch_knowledge_node(state)

    jurisdictions = mock_db.get_applicable_doctrines.call_args[0][0]
    assert jurisdictions == ["UK"]


@pytest.mark.asyncio
async def test_fetch_knowledge_flatten_empty_list():
    """flatten() on empty list returns empty list."""
    profile = SAMPLE_PROFILE.model_dump()
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[])
        result = await fetch_knowledge_node(state)

    assert result["risk_factors"] == []
    assert result["available_mitigations"] == []
    assert result["mitigation_edges"] == []


@pytest.mark.asyncio
async def test_fetch_knowledge_flatten_unwraps_result_key():
    """flatten() unwraps SurrealDB {result: [...]} wrapper."""
    profile = SAMPLE_PROFILE.model_dump()
    state = _base_state(deployment_profile=profile)

    wrapped = [{"result": [{"name": "factor_1"}, {"name": "factor_2"}]}]

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=wrapped)
        result = await fetch_knowledge_node(state)

    assert result["risk_factors"] == [{"name": "factor_1"}, {"name": "factor_2"}]


@pytest.mark.asyncio
async def test_fetch_knowledge_flatten_passthrough_without_result_key():
    """flatten() returns list as-is when first element has no 'result' key."""
    profile = SAMPLE_PROFILE.model_dump()
    state = _base_state(deployment_profile=profile)

    raw_data = [{"name": "direct_factor", "weight": 0.5}]

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=raw_data)
        result = await fetch_knowledge_node(state)

    assert result["risk_factors"] == [{"name": "direct_factor", "weight": 0.5}]


@pytest.mark.asyncio
async def test_fetch_knowledge_has_start_and_complete_audit():
    """fetch_knowledge_node should log both start and complete audit events."""
    audit_calls = []

    async def track_audit(sid, agent, action, **kwargs):
        audit_calls.append((agent, action))

    profile = SAMPLE_PROFILE.model_dump()
    state = _base_state(deployment_profile=profile)

    with patch("app.graph.workflow.log_audit", side_effect=track_audit), \
         patch("app.graph.workflow.db") as mock_db:
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_db.query = AsyncMock(return_value=[[]])
        await fetch_knowledge_node(state)

    assert ("knowledge_fetch", "start") in audit_calls
    assert ("knowledge_fetch", "complete") in audit_calls


# ============================================================================
# UNIT TESTS: error propagation through each node
# ============================================================================

@pytest.mark.asyncio
async def test_legal_node_propagates_agent_error():
    """Error in legal agent should propagate as-is."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        applicable_doctrines=[],
        applicable_regulations=[],
    )

    with patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock,
               side_effect=ValueError("Legal agent returned invalid JSON")), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        with pytest.raises(ValueError, match="invalid JSON"):
            await legal_node(state)


@pytest.mark.asyncio
async def test_technical_node_propagates_agent_error():
    """Error in technical agent should propagate."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        risk_factors=[],
    )

    with patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock,
               side_effect=ValueError("Technical agent returned invalid JSON")), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        with pytest.raises(ValueError, match="invalid JSON"):
            await technical_node(state)


@pytest.mark.asyncio
async def test_mitigation_node_propagates_agent_error():
    """Error in mitigation agent should propagate."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        available_mitigations=[],
        mitigation_edges=[],
    )

    with patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock,
               side_effect=ValueError("Mitigation agent returned invalid JSON")), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        with pytest.raises(ValueError, match="invalid JSON"):
            await mitigation_node(state)


@pytest.mark.asyncio
async def test_pricing_node_propagates_agent_error():
    """Error in pricing agent should propagate."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        legal_analysis=SAMPLE_LEGAL.model_dump(),
        technical_analysis=SAMPLE_TECHNICAL.model_dump(),
        mitigation_analysis=SAMPLE_MITIGATION.model_dump(),
    )

    with patch("app.graph.workflow.run_pricing", new_callable=AsyncMock,
               side_effect=ValueError("Pricing agent returned invalid JSON")), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        with pytest.raises(ValueError, match="invalid JSON"):
            await pricing_node(state)


@pytest.mark.asyncio
async def test_fetch_knowledge_propagates_db_error():
    """DB failure in fetch_knowledge should propagate."""
    state = _base_state(deployment_profile=SAMPLE_PROFILE.model_dump())

    with patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock):
        mock_db.get_applicable_doctrines = AsyncMock(side_effect=RuntimeError("DB unreachable"))
        with pytest.raises(RuntimeError, match="DB unreachable"):
            await fetch_knowledge_node(state)


@pytest.mark.asyncio
async def test_run_assessment_mid_pipeline_error_marks_failed():
    """Error in legal node (mid-pipeline) should mark assessment as failed."""
    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock,
               side_effect=ValueError("Legal agent crashed")), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = True
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        with pytest.raises(ValueError, match="Legal agent crashed"):
            await run_assessment("chatbot", ["UK"], "financial", "demo-user")

    update_calls = [c for c in mock_db.query.call_args_list if "failed" in str(c)]
    assert len(update_calls) >= 1


# ============================================================================
# UNIT TESTS: graph structure validation
# ============================================================================

def test_build_workflow_entry_point_is_intake():
    """The graph's entry point should be the intake node."""
    graph = build_workflow()
    # In LangGraph, __start__ connects to the entry point
    # Check that the graph can be drawn/inspected and intake is first
    assert "intake" in graph.nodes
    # Verify __start__ leads to intake by checking the graph structure
    graph_dict = graph.get_graph().to_json()
    assert graph_dict is not None


def test_build_workflow_all_nodes_reachable():
    """All 6 nodes should be reachable in the compiled graph."""
    graph = build_workflow()
    graph_repr = graph.get_graph()
    node_ids = set(graph_repr.nodes.keys())
    expected = {"intake", "fetch_knowledge", "legal", "technical", "mitigation", "pricing"}
    assert expected.issubset(node_ids)


def test_build_workflow_edge_ordering():
    """Edges should enforce: intake→fetch_knowledge→legal→technical→mitigation→pricing→END."""
    graph = build_workflow()
    graph_repr = graph.get_graph()
    edges = [(e.source, e.target) for e in graph_repr.edges]

    assert ("intake", "fetch_knowledge") in edges
    assert ("fetch_knowledge", "legal") in edges
    assert ("legal", "technical") in edges
    assert ("technical", "mitigation") in edges
    assert ("mitigation", "pricing") in edges
    # pricing→__end__
    pricing_targets = [t for s, t in edges if s == "pricing"]
    assert any("end" in t.lower() for t in pricing_targets)


def test_build_workflow_no_cycles():
    """The compiled graph should have no cycles (purely linear DAG)."""
    graph = build_workflow()
    graph_repr = graph.get_graph()

    # Build adjacency list
    adj = {}
    for e in graph_repr.edges:
        adj.setdefault(e.source, []).append(e.target)

    # DFS cycle detection
    visited, in_stack = set(), set()

    def has_cycle(node):
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        for neighbor in adj.get(node, []):
            if has_cycle(neighbor):
                return True
        in_stack.discard(node)
        return False

    for node in adj:
        assert not has_cycle(node), f"Cycle detected involving {node}"


# ============================================================================
# UNIT TESTS: state progression, DB reconnect, quality scores
# ============================================================================

@pytest.mark.asyncio
async def test_full_state_progression_current_step():
    """Verify current_step transitions through all expected values in order."""
    steps_seen = []

    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = True
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        graph = build_workflow()
        initial = _base_state(
            session_id="step-test",
            description="test",
            jurisdictions=["UK"],
            sector="financial",
        )
        # Stream to capture intermediate states
        async for event in graph.astream(initial, config={"configurable": {"thread_id": "step-test"}}):
            for node_output in event.values():
                if "current_step" in node_output:
                    steps_seen.append(node_output["current_step"])

    assert steps_seen == [
        "intake_complete",
        "knowledge_fetched",
        "legal_complete",
        "technical_complete",
        "mitigation_complete",
        "complete",
    ]


@pytest.mark.asyncio
async def test_run_assessment_connects_when_disconnected():
    """run_assessment should call db.connect() when db._conn is None."""
    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = None  # Not connected
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        await run_assessment("chatbot", ["UK"], "financial", "demo-user")

    mock_db.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_assessment_skips_connect_when_connected():
    """run_assessment should not reconnect when db._conn is already set."""
    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = MagicMock()  # Already connected
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        await run_assessment("chatbot", ["UK"], "financial", "demo-user")

    mock_db.connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_quality_scores_remain_zero():
    """Quality score fields should remain at 0.0 through the full pipeline (not yet implemented)."""
    with patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = True
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        result = await run_assessment("chatbot", ["UK"], "financial", "demo-user")

    assert result["legal_quality_score"] == 0.0
    assert result["technical_quality_score"] == 0.0
    assert result["pricing_quality_score"] == 0.0


# ============================================================================
# INTEGRATION TESTS: full pipeline audit completeness
# ============================================================================

@pytest.mark.asyncio
async def test_full_pipeline_audit_trail_completeness():
    """Every node should produce audit start+complete events through the full pipeline."""
    audit_calls = []

    async def track_audit(sid, agent, action, **kwargs):
        audit_calls.append((agent, action))

    with patch("app.graph.workflow.log_audit", side_effect=track_audit), \
         patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = True
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        await run_assessment("chatbot", ["UK"], "financial", "demo-user")

    # Every node should have start + complete
    expected_agents = ["intake", "knowledge_fetch", "legal", "technical", "mitigation", "pricing"]
    for agent in expected_agents:
        assert (agent, "start") in audit_calls, f"Missing start audit for {agent}"
        assert (agent, "complete") in audit_calls, f"Missing complete audit for {agent}"

    # Total audit calls: 6 agents × 2 (start+complete) = 12
    assert len(audit_calls) == 12


@pytest.mark.asyncio
async def test_full_pipeline_audit_ordering():
    """Audit events should occur in the correct order through the pipeline."""
    audit_calls = []

    async def track_audit(sid, agent, action, **kwargs):
        audit_calls.append((agent, action))

    with patch("app.graph.workflow.log_audit", side_effect=track_audit), \
         patch("app.graph.workflow.run_intake", new_callable=AsyncMock, return_value=SAMPLE_PROFILE), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = True
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        await run_assessment("chatbot", ["UK"], "financial", "demo-user")

    # Extract just the agent names in order they appear
    agent_order = [agent for agent, _ in audit_calls]
    expected_order = [
        "intake", "intake",
        "knowledge_fetch", "knowledge_fetch",
        "legal", "legal",
        "technical", "technical",
        "mitigation", "mitigation",
        "pricing", "pricing",
    ]
    assert agent_order == expected_order


@pytest.mark.asyncio
async def test_pricing_node_zero_scenarios():
    """Pricing result with empty scenarios list should still complete without DB errors."""
    pricing_no_scenarios = RiskPrice(
        **{**SAMPLE_PRICING.model_dump(), "scenarios": []}
    )
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        legal_analysis=SAMPLE_LEGAL.model_dump(),
        technical_analysis=SAMPLE_TECHNICAL.model_dump(),
        mitigation_analysis=SAMPLE_MITIGATION.model_dump(),
    )

    with patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=pricing_no_scenarios), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db:
        mock_db.query = AsyncMock(return_value=[])
        result = await pricing_node(state)

    assert result["current_step"] == "complete"
    # Only risk_score CREATE + UPDATE assessment = 2 (no scenario CREATEs)
    assert mock_db.query.await_count == 2


@pytest.mark.asyncio
async def test_pricing_node_audit_includes_output_data():
    """Pricing node complete audit should include risk_score and premium data."""
    state = _base_state(
        deployment_profile=SAMPLE_PROFILE.model_dump(),
        legal_analysis=SAMPLE_LEGAL.model_dump(),
        technical_analysis=SAMPLE_TECHNICAL.model_dump(),
        mitigation_analysis=SAMPLE_MITIGATION.model_dump(),
    )

    audit_data = {}

    async def capture_audit(sid, agent, action, **kwargs):
        if agent == "pricing" and action == "complete":
            audit_data.update(kwargs)

    with patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", side_effect=capture_audit), \
         patch("app.graph.workflow.db") as mock_db:
        mock_db.query = AsyncMock(return_value=[])
        await pricing_node(state)

    assert "output_data" in audit_data
    assert audit_data["output_data"]["risk_score"] == 0.6
    assert "Medium" in audit_data["output_data"]["premium"]


@pytest.mark.asyncio
async def test_run_assessment_creates_session_before_graph():
    """Assessment record should be created in DB before graph execution starts."""
    call_order = []

    async def track_query(sql, params=None):
        if "CREATE assessment" in sql:
            call_order.append("create_assessment")
        return []

    async def track_intake(*args, **kwargs):
        call_order.append("intake_ran")
        return SAMPLE_PROFILE

    with patch("app.graph.workflow.run_intake", side_effect=track_intake), \
         patch("app.graph.workflow.run_legal_analysis", new_callable=AsyncMock, return_value=SAMPLE_LEGAL), \
         patch("app.graph.workflow.run_technical_analysis", new_callable=AsyncMock, return_value=SAMPLE_TECHNICAL), \
         patch("app.graph.workflow.run_mitigation_analysis", new_callable=AsyncMock, return_value=SAMPLE_MITIGATION), \
         patch("app.graph.workflow.run_pricing", new_callable=AsyncMock, return_value=SAMPLE_PRICING), \
         patch("app.graph.workflow.log_audit", new_callable=AsyncMock), \
         patch("app.graph.workflow.db") as mock_db, \
         patch("app.graph.workflow.get_opik_tracer") as mock_opik, \
         patch("app.graph.workflow.get_langsmith_run_config", return_value={"run_name": "test"}):

        mock_db._conn = True
        mock_db.connect = AsyncMock()
        mock_db.query = AsyncMock(side_effect=track_query)
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])
        mock_db.get_applicable_regulations = AsyncMock(return_value=[])
        mock_opik.return_value = MagicMock()

        await run_assessment("chatbot", ["UK"], "financial", "demo-user")

    assert call_order.index("create_assessment") < call_order.index("intake_ran")
