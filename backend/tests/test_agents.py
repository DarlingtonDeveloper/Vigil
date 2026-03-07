"""Tests for SPEC-04: LangGraph Agents.

All LLM calls are mocked — these tests verify:
- Pydantic models validate correctly
- Agents parse JSON from LLM responses (including code-fenced)
- Agents raise ValueError on invalid JSON
- Agents raise ValueError on schema-invalid JSON
- No agent writes to SurrealDB
"""
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agents.intake import (
    DeploymentProfile,
    ToolDescription,
    VendorInfo,
    run_intake,
    _extract_json,
)
from app.agents.legal import (
    LegalAnalysis,
    DoctrineAssessment,
    RegulatoryGap,
    run_legal_analysis,
)
from app.agents.technical import (
    TechnicalAnalysis,
    FactorScore,
    run_technical_analysis,
)
from app.agents.mitigation import (
    MitigationAnalysis,
    MitigationAxisScore,
    MitigationScore,
    run_mitigation_analysis,
)
from app.agents.pricing import (
    RiskPrice,
    RiskScenario,
    run_pricing,
)


# ---------------------------------------------------------------------------
# Fixtures: valid model data
# ---------------------------------------------------------------------------

VALID_DEPLOYMENT_PROFILE = {
    "agent_description": "Customer support chatbot",
    "tools": [
        {
            "name": "search_kb",
            "action_type": "read",
            "description": "Searches knowledge base",
            "risk_note": None,
        }
    ],
    "data_access": ["PII", "internal"],
    "autonomy_level": "human_on_the_loop",
    "output_reach": "customer_facing",
    "sector": "financial",
    "jurisdictions": ["UK", "EU"],
    "human_oversight_model": "Team lead reviews flagged responses",
    "reviewer_qualification": "domain_expert",
    "existing_guardrails": ["content filter", "PII redaction"],
    "vendor_info": {
        "provider": "Anthropic",
        "model": "claude-sonnet-4-20250514",
        "indemnification": "partial",
        "contract_reviewed": True,
    },
    "key_risks_identified": ["PII exposure", "hallucinated financial advice"],
}

VALID_LEGAL_ANALYSIS = {
    "legal_exposure_score": 0.65,
    "doctrine_assessments": [
        {
            "doctrine_name": "Vicarious Liability",
            "applies": True,
            "exposure_level": "high",
            "reasoning": "Agent acts on behalf of company",
            "worst_case": "Company liable for all agent actions",
        }
    ],
    "regulatory_gaps": [
        {
            "regulation": "EU AI Act",
            "requirement": "Human oversight",
            "status": "partial",
            "risk_if_non_compliant": "Fines up to 35M EUR",
        }
    ],
    "contract_formation_risk": "Medium — agent can make promises to customers",
    "tort_exposure": "High — financial advice context",
    "key_uncertainties": ["Novel litigation risk for AI advisors"],
    "confidence": 0.7,
}

VALID_TECHNICAL_ANALYSIS = {
    "technical_risk_score": 0.55,
    "factor_scores": [
        {
            "factor_name": "autonomy_level",
            "level": "human_on_the_loop",
            "score": 0.6,
            "reasoning": "Agent acts with monitoring but no pre-approval",
            "missing_info": None,
        }
    ],
    "amplification_effects": ["Customer-facing + financial sector = compounded exposure"],
    "key_vulnerabilities": ["No confidence thresholds on outputs"],
    "confidence": 0.75,
}

VALID_MITIGATION_ANALYSIS = {
    "overall_mitigation_score": 0.45,
    "axis_scores": [
        {
            "axis": "architectural",
            "score": 0.6,
            "present_mitigations": ["content filter"],
            "missing_mitigations": ["confidence thresholds"],
            "critical_gaps": ["No output schema enforcement"],
        }
    ],
    "recommendations": [
        {
            "name": "Add confidence thresholds",
            "priority": "high",
            "impact": "0.15 risk reduction",
            "cost": "moderate",
            "reasoning": "Prevents low-confidence outputs reaching customers",
        }
    ],
    "quick_wins": ["Add output logging"],
    "confidence": 0.7,
}

VALID_RISK_PRICE = {
    "executive_summary": "High-risk deployment requiring additional controls",
    "overall_risk_score": 0.62,
    "technical_risk": 0.55,
    "legal_exposure": 0.65,
    "mitigation_effectiveness": 1.45,
    "premium_band": "high ($50K-$200K/yr)",
    "premium_reasoning": "Financial sector + customer-facing + partial compliance",
    "top_exposures": [
        {
            "exposure": "Hallucinated financial advice",
            "severity": "major",
            "mitigation_available": True,
        }
    ],
    "scenarios": [
        {
            "scenario_type": "Incorrect financial guidance",
            "probability": "possible",
            "severity": "major",
            "expected_loss_range": "$100K-$1M",
            "applicable_doctrines": ["Negligent Misstatement"],
            "mitigation_options": ["Confidence thresholds", "Expert review"],
        }
    ],
    "recommendations": [
        {
            "action": "Implement confidence thresholds",
            "priority": "critical",
            "impact": "0.15 risk reduction",
            "reasoning": "Prevents high-risk outputs",
        }
    ],
    "confidence": 0.7,
    "data_gaps": ["No incident response plan documented"],
}


# ---------------------------------------------------------------------------
# Helper to build a mock LLM response
# ---------------------------------------------------------------------------

def _mock_llm_response(data: dict, code_fenced: bool = False) -> MagicMock:
    """Create a mock LLM response with .content as a JSON string."""
    raw = json.dumps(data)
    if code_fenced:
        raw = f"```json\n{raw}\n```"
    resp = MagicMock()
    resp.content = raw
    resp.usage_metadata = None
    return resp


# ---------------------------------------------------------------------------
# Test: _extract_json helper
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_code_fenced_json(self):
        raw = '```json\n{"a": 1}\n```'
        assert _extract_json(raw) == '{"a": 1}'

    def test_generic_code_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert _extract_json(raw) == '{"a": 1}'

    def test_text_around_fence(self):
        raw = 'Here is the result:\n```json\n{"a": 1}\n```\nDone.'
        assert _extract_json(raw) == '{"a": 1}'


# ---------------------------------------------------------------------------
# Test: Pydantic models validate correctly
# ---------------------------------------------------------------------------

class TestModels:
    def test_deployment_profile(self):
        dp = DeploymentProfile(**VALID_DEPLOYMENT_PROFILE)
        assert dp.agent_description == "Customer support chatbot"
        assert len(dp.tools) == 1
        assert dp.tools[0].action_type == "read"
        assert dp.vendor_info is not None
        assert dp.vendor_info.provider == "Anthropic"

    def test_deployment_profile_no_vendor(self):
        data = {**VALID_DEPLOYMENT_PROFILE, "vendor_info": None}
        dp = DeploymentProfile(**data)
        assert dp.vendor_info is None

    def test_tool_description(self):
        t = ToolDescription(name="x", action_type="write", description="y")
        assert t.risk_note is None

    def test_vendor_info(self):
        v = VendorInfo(provider="OpenAI", model="gpt-4o", indemnification="none", contract_reviewed=False)
        assert v.contract_reviewed is False

    def test_legal_analysis(self):
        la = LegalAnalysis(**VALID_LEGAL_ANALYSIS)
        assert 0.0 <= la.legal_exposure_score <= 1.0
        assert len(la.doctrine_assessments) == 1
        assert la.doctrine_assessments[0].applies is True

    def test_doctrine_assessment(self):
        da = DoctrineAssessment(
            doctrine_name="test", applies=False, exposure_level="low",
            reasoning="n/a", worst_case="n/a",
        )
        assert da.applies is False

    def test_regulatory_gap(self):
        rg = RegulatoryGap(
            regulation="GDPR", requirement="DPIA", status="compliant",
            risk_if_non_compliant="fines",
        )
        assert rg.status == "compliant"

    def test_technical_analysis(self):
        ta = TechnicalAnalysis(**VALID_TECHNICAL_ANALYSIS)
        assert 0.0 <= ta.technical_risk_score <= 1.0
        assert len(ta.factor_scores) == 1

    def test_factor_score(self):
        fs = FactorScore(
            factor_name="x", level="high", score=0.9, reasoning="y",
        )
        assert fs.missing_info is None

    def test_mitigation_analysis(self):
        ma = MitigationAnalysis(**VALID_MITIGATION_ANALYSIS)
        assert 0.0 <= ma.overall_mitigation_score <= 1.0
        assert len(ma.quick_wins) == 1

    def test_mitigation_axis_score(self):
        mas = MitigationAxisScore(
            axis="human_oversight", score=0.3,
            present_mitigations=[], missing_mitigations=["a"],
            critical_gaps=[],
        )
        assert mas.score == 0.3

    def test_mitigation_score(self):
        ms = MitigationScore(
            mitigation_name="audit", present=True,
            effectiveness_if_present=0.8, notes="good",
        )
        assert ms.present is True

    def test_risk_price(self):
        rp = RiskPrice(**VALID_RISK_PRICE)
        assert 0.0 <= rp.overall_risk_score <= 1.0
        assert len(rp.scenarios) == 1

    def test_risk_scenario(self):
        rs = RiskScenario(
            scenario_type="breach", probability="possible",
            severity="major", expected_loss_range="$10K-$50K",
            applicable_doctrines=["x"], mitigation_options=["y"],
        )
        assert rs.probability == "possible"


# ---------------------------------------------------------------------------
# Test: run_intake
# ---------------------------------------------------------------------------

class TestRunIntake:
    @pytest.mark.asyncio
    async def test_returns_deployment_profile(self):
        mock_resp = _mock_llm_response(VALID_DEPLOYMENT_PROFILE)
        with patch("app.agents.intake.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_intake("test description", ["UK"], "financial", "sess-1")
        assert isinstance(result, DeploymentProfile)
        assert result.sector == "financial"

    @pytest.mark.asyncio
    async def test_handles_code_fenced_json(self):
        mock_resp = _mock_llm_response(VALID_DEPLOYMENT_PROFILE, code_fenced=True)
        with patch("app.agents.intake.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_intake("test", ["EU"], None, "sess-2")
        assert isinstance(result, DeploymentProfile)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.content = "not json at all"
        mock_resp.usage_metadata = None
        with patch("app.agents.intake.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="invalid JSON"):
                await run_intake("test", ["UK"], None, "sess-3")

    @pytest.mark.asyncio
    async def test_raises_on_schema_mismatch(self):
        mock_resp = _mock_llm_response({"wrong": "schema"})
        with patch("app.agents.intake.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="failed validation"):
                await run_intake("test", ["UK"], None, "sess-4")


# ---------------------------------------------------------------------------
# Test: run_legal_analysis
# ---------------------------------------------------------------------------

class TestRunLegalAnalysis:
    @pytest.mark.asyncio
    async def test_returns_legal_analysis(self):
        mock_resp = _mock_llm_response(VALID_LEGAL_ANALYSIS)
        with patch("app.agents.legal.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_legal_analysis(
                VALID_DEPLOYMENT_PROFILE,
                [{"name": "Vicarious Liability"}],
                [{"name": "EU AI Act"}],
                "sess-1",
            )
        assert isinstance(result, LegalAnalysis)
        assert result.legal_exposure_score == 0.65

    @pytest.mark.asyncio
    async def test_handles_code_fenced_json(self):
        mock_resp = _mock_llm_response(VALID_LEGAL_ANALYSIS, code_fenced=True)
        with patch("app.agents.legal.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_legal_analysis({}, [], [], "sess-2")
        assert isinstance(result, LegalAnalysis)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.content = "broken {json"
        mock_resp.usage_metadata = None
        with patch("app.agents.legal.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="invalid JSON"):
                await run_legal_analysis({}, [], [], "sess-3")

    @pytest.mark.asyncio
    async def test_raises_on_schema_mismatch(self):
        mock_resp = _mock_llm_response({"not": "right"})
        with patch("app.agents.legal.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="failed validation"):
                await run_legal_analysis({}, [], [], "sess-4")


# ---------------------------------------------------------------------------
# Test: run_technical_analysis
# ---------------------------------------------------------------------------

class TestRunTechnicalAnalysis:
    @pytest.mark.asyncio
    async def test_returns_technical_analysis(self):
        mock_resp = _mock_llm_response(VALID_TECHNICAL_ANALYSIS)
        with patch("app.agents.technical.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_technical_analysis(
                VALID_DEPLOYMENT_PROFILE,
                [{"name": "autonomy_level", "levels": []}],
                "sess-1",
            )
        assert isinstance(result, TechnicalAnalysis)
        assert result.technical_risk_score == 0.55

    @pytest.mark.asyncio
    async def test_handles_code_fenced_json(self):
        mock_resp = _mock_llm_response(VALID_TECHNICAL_ANALYSIS, code_fenced=True)
        with patch("app.agents.technical.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_technical_analysis({}, [], "sess-2")
        assert isinstance(result, TechnicalAnalysis)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.content = ""
        mock_resp.usage_metadata = None
        with patch("app.agents.technical.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="invalid JSON"):
                await run_technical_analysis({}, [], "sess-3")

    @pytest.mark.asyncio
    async def test_raises_on_schema_mismatch(self):
        mock_resp = _mock_llm_response({"bad": "data"})
        with patch("app.agents.technical.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="failed validation"):
                await run_technical_analysis({}, [], "sess-4")


# ---------------------------------------------------------------------------
# Test: run_mitigation_analysis
# ---------------------------------------------------------------------------

class TestRunMitigationAnalysis:
    @pytest.mark.asyncio
    async def test_returns_mitigation_analysis(self):
        mock_resp = _mock_llm_response(VALID_MITIGATION_ANALYSIS)
        with patch("app.agents.mitigation.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_mitigation_analysis(
                VALID_DEPLOYMENT_PROFILE, [], [], "sess-1",
            )
        assert isinstance(result, MitigationAnalysis)
        assert result.overall_mitigation_score == 0.45

    @pytest.mark.asyncio
    async def test_handles_code_fenced_json(self):
        mock_resp = _mock_llm_response(VALID_MITIGATION_ANALYSIS, code_fenced=True)
        with patch("app.agents.mitigation.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_mitigation_analysis({}, [], [], "sess-2")
        assert isinstance(result, MitigationAnalysis)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.content = "```json\n{invalid}\n```"
        mock_resp.usage_metadata = None
        with patch("app.agents.mitigation.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="invalid JSON"):
                await run_mitigation_analysis({}, [], [], "sess-3")

    @pytest.mark.asyncio
    async def test_raises_on_schema_mismatch(self):
        mock_resp = _mock_llm_response({"missing": "fields"})
        with patch("app.agents.mitigation.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="failed validation"):
                await run_mitigation_analysis({}, [], [], "sess-4")


# ---------------------------------------------------------------------------
# Test: run_pricing
# ---------------------------------------------------------------------------

class TestRunPricing:
    @pytest.mark.asyncio
    async def test_returns_risk_price(self):
        mock_resp = _mock_llm_response(VALID_RISK_PRICE)
        with patch("app.agents.pricing.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_pricing(
                VALID_LEGAL_ANALYSIS,
                VALID_TECHNICAL_ANALYSIS,
                VALID_MITIGATION_ANALYSIS,
                VALID_DEPLOYMENT_PROFILE,
                "sess-1",
            )
        assert isinstance(result, RiskPrice)
        assert result.overall_risk_score == 0.62
        assert "high" in result.premium_band

    @pytest.mark.asyncio
    async def test_handles_code_fenced_json(self):
        mock_resp = _mock_llm_response(VALID_RISK_PRICE, code_fenced=True)
        with patch("app.agents.pricing.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            result = await run_pricing({}, {}, {}, {}, "sess-2")
        assert isinstance(result, RiskPrice)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_json(self):
        mock_resp = MagicMock()
        mock_resp.content = "I cannot produce JSON"
        mock_resp.usage_metadata = None
        with patch("app.agents.pricing.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="invalid JSON"):
                await run_pricing({}, {}, {}, {}, "sess-3")

    @pytest.mark.asyncio
    async def test_raises_on_schema_mismatch(self):
        mock_resp = _mock_llm_response({"only": "one_field"})
        with patch("app.agents.pricing.ChatAnthropic") as MockLLM:
            MockLLM.return_value.ainvoke = AsyncMock(return_value=mock_resp)
            with pytest.raises(ValueError, match="failed validation"):
                await run_pricing({}, {}, {}, {}, "sess-4")


# ---------------------------------------------------------------------------
# Test: No agent writes to SurrealDB
# ---------------------------------------------------------------------------

class TestNoSurrealDBWrites:
    """Verify that no agent module imports or uses the SurrealDB client."""

    def _assert_no_db_usage(self, module):
        source = open(module.__file__).read()
        # Check that there are no import statements for surrealdb or db client
        assert "import surrealdb" not in source
        assert "from surrealdb" not in source
        assert "from app.db" not in source
        assert "import app.db" not in source

    def test_intake_no_db_import(self):
        import app.agents.intake as m
        self._assert_no_db_usage(m)

    def test_legal_no_db_import(self):
        import app.agents.legal as m
        self._assert_no_db_usage(m)

    def test_technical_no_db_import(self):
        import app.agents.technical as m
        self._assert_no_db_usage(m)

    def test_mitigation_no_db_import(self):
        import app.agents.mitigation as m
        self._assert_no_db_usage(m)

    def test_pricing_no_db_import(self):
        import app.agents.pricing as m
        self._assert_no_db_usage(m)
