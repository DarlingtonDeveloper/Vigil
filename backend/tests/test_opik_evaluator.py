"""Tests for Opik quality gate evaluators."""
import pytest
from unittest.mock import AsyncMock, patch

from app.tracing.opik_evaluator import (
    EvalResult,
    QUALITY_THRESHOLD,
    validate_legal_analysis,
    validate_technical_analysis,
    validate_pricing,
    auto_evaluate_session,
)


@pytest.fixture(autouse=True)
def mock_db_query():
    """Mock SurrealDB writes for all evaluator tests."""
    with patch("app.tracing.opik_evaluator.db") as mock_db:
        mock_db.query = AsyncMock(return_value=None)
        yield mock_db


# --- EvalResult model ---


def test_eval_result_model():
    result = EvalResult(stage="test", passed=True, score=0.9, reasons=["ok"])
    assert result.stage == "test"
    assert result.passed is True
    assert result.score == 0.9
    assert result.reasons == ["ok"]


# --- validate_legal_analysis ---


@pytest.mark.asyncio
async def test_legal_valid_analysis_passes():
    analysis = {
        "doctrine_assessments": [
            {"doctrine_name": "negligence"},
            {"doctrine_name": "vicarious_liability"},
        ],
        "legal_exposure_score": 0.6,
        "confidence": 0.8,
    }
    profile = {"tools": [], "data_access": []}
    known = ["negligence", "vicarious_liability", "strict_liability"]

    result = await validate_legal_analysis(analysis, profile, known, "sess-1")
    assert result.passed is True
    assert result.score == 1.0
    assert result.reasons == ["Passed all checks"]


@pytest.mark.asyncio
async def test_legal_unknown_doctrine_penalized():
    analysis = {
        "doctrine_assessments": [
            {"doctrine_name": "made_up_doctrine"},
        ],
        "legal_exposure_score": 0.5,
        "confidence": 0.7,
    }
    profile = {"tools": [], "data_access": []}
    known = ["negligence"]

    result = await validate_legal_analysis(analysis, profile, known, "sess-1")
    assert result.score == pytest.approx(0.7)
    assert any("Unknown doctrine" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_legal_exposure_out_of_range():
    analysis = {
        "doctrine_assessments": [],
        "legal_exposure_score": 1.5,
        "confidence": 0.7,
    }
    profile = {"tools": [], "data_access": []}

    result = await validate_legal_analysis(analysis, profile, [], "sess-1")
    assert result.score == pytest.approx(0.5)
    assert any("out of range" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_legal_overconfident_penalized():
    analysis = {
        "doctrine_assessments": [],
        "legal_exposure_score": 0.5,
        "confidence": 0.99,
    }
    profile = {"tools": [], "data_access": []}

    result = await validate_legal_analysis(analysis, profile, [], "sess-1")
    assert result.score == pytest.approx(0.8)
    assert any("overconfident" in r.lower() for r in result.reasons)


@pytest.mark.asyncio
async def test_legal_external_comms_needs_apparent_authority():
    analysis = {
        "doctrine_assessments": [{"doctrine_name": "negligence"}],
        "legal_exposure_score": 0.5,
        "confidence": 0.7,
    }
    profile = {
        "tools": [{"action_type": "communicate"}],
        "data_access": [],
    }
    known = ["negligence", "apparent_authority"]

    result = await validate_legal_analysis(analysis, profile, known, "sess-1")
    assert result.score == pytest.approx(0.8)
    assert any("apparent authority" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_legal_pii_needs_gdpr():
    analysis = {
        "doctrine_assessments": [],
        "legal_exposure_score": 0.5,
        "confidence": 0.7,
        "regulatory_gaps": [],
    }
    profile = {"tools": [], "data_access": ["PII"]}

    result = await validate_legal_analysis(analysis, profile, [], "sess-1")
    assert result.score == pytest.approx(0.85)
    assert any("GDPR" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_legal_multiple_penalties_stack():
    analysis = {
        "doctrine_assessments": [{"doctrine_name": "fake_1"}, {"doctrine_name": "fake_2"}],
        "legal_exposure_score": -0.1,
        "confidence": 0.99,
    }
    profile = {"tools": [], "data_access": []}

    result = await validate_legal_analysis(analysis, profile, [], "sess-1")
    # -0.3 * 2 (unknown) - 0.5 (range) - 0.2 (confidence) = -0.4
    assert result.passed is False
    assert len(result.reasons) >= 3


@pytest.mark.asyncio
async def test_legal_logs_to_surrealdb(mock_db_query):
    analysis = {
        "doctrine_assessments": [],
        "legal_exposure_score": 0.5,
        "confidence": 0.7,
    }
    profile = {"tools": [], "data_access": []}

    await validate_legal_analysis(analysis, profile, [], "sess-1")
    mock_db_query.query.assert_called_once()
    call_args = mock_db_query.query.call_args
    assert "CREATE evaluation" in call_args[0][0]
    assert call_args[0][1]["sid"] == "sess-1"
    assert call_args[0][1]["score"] == 1.0


# --- validate_technical_analysis ---


@pytest.mark.asyncio
async def test_technical_valid_analysis_passes():
    risk_factors = [{"name": "autonomy"}, {"name": "opacity"}]
    analysis = {
        "factor_scores": [
            {"factor_name": "autonomy", "score": 0.7},
            {"factor_name": "opacity", "score": 0.5},
        ],
        "technical_risk_score": 0.6,
        "amplification_effects": [],
    }

    result = await validate_technical_analysis(analysis, risk_factors, "sess-1")
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_technical_missing_factor_scores():
    risk_factors = [{"name": "autonomy"}, {"name": "opacity"}, {"name": "reach"}]
    analysis = {
        "factor_scores": [{"factor_name": "autonomy", "score": 0.5}],
        "technical_risk_score": 0.5,
        "amplification_effects": [],
    }

    result = await validate_technical_analysis(analysis, risk_factors, "sess-1")
    # 2 missing * 0.1 = 0.2 penalty
    assert result.score == pytest.approx(0.8)
    assert any("Missing scores" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_technical_score_out_of_range():
    risk_factors = [{"name": "autonomy"}]
    analysis = {
        "factor_scores": [{"factor_name": "autonomy", "score": 1.5}],
        "technical_risk_score": 0.5,
        "amplification_effects": [],
    }

    result = await validate_technical_analysis(analysis, risk_factors, "sess-1")
    assert result.score == pytest.approx(0.8)
    assert any("out of range" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_technical_overall_score_out_of_range():
    risk_factors = []
    analysis = {
        "factor_scores": [],
        "technical_risk_score": 1.5,
        "amplification_effects": [],
    }

    result = await validate_technical_analysis(analysis, risk_factors, "sess-1")
    assert result.score == pytest.approx(0.5)
    assert result.passed is True  # exactly at threshold


@pytest.mark.asyncio
async def test_technical_amplification_bad_reference():
    risk_factors = [{"name": "autonomy"}]
    analysis = {
        "factor_scores": [{"factor_name": "autonomy", "score": 0.5}],
        "technical_risk_score": 0.5,
        "amplification_effects": ["Some unrelated effect that mentions nothing"],
    }

    result = await validate_technical_analysis(analysis, risk_factors, "sess-1")
    assert result.score == pytest.approx(0.9)
    assert any("Amplification" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_technical_logs_to_surrealdb(mock_db_query):
    analysis = {
        "factor_scores": [],
        "technical_risk_score": 0.5,
        "amplification_effects": [],
    }

    await validate_technical_analysis(analysis, [], "sess-1")
    mock_db_query.query.assert_called_once()
    call_args = mock_db_query.query.call_args
    assert "opik_gate_technical" in call_args[0][0]


# --- validate_pricing ---


@pytest.mark.asyncio
async def test_pricing_valid_passes():
    pricing = {
        "overall_risk_score": 0.5,
        "premium_band": "Medium",
        "scenarios": [{"name": "s1"}],
        "recommendations": [{"text": "r1"}],
        "confidence": 0.75,
    }

    result = await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    assert result.passed is True
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_pricing_risk_score_out_of_range():
    pricing = {
        "overall_risk_score": 1.2,
        "premium_band": "High",
        "scenarios": [{"name": "s1"}],
        "recommendations": [{"text": "r1"}],
        "confidence": 0.7,
    }

    result = await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    assert result.score == pytest.approx(0.5)
    assert any("out of range" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_pricing_low_risk_high_premium_inconsistent():
    pricing = {
        "overall_risk_score": 0.2,
        "premium_band": "High Risk",
        "scenarios": [{"name": "s1"}],
        "recommendations": [{"text": "r1"}],
        "confidence": 0.7,
    }

    result = await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    assert result.score == pytest.approx(0.7)
    assert any("inconsistent" in r.lower() for r in result.reasons)


@pytest.mark.asyncio
async def test_pricing_high_risk_low_premium_inconsistent():
    pricing = {
        "overall_risk_score": 0.8,
        "premium_band": "Low Risk",
        "scenarios": [{"name": "s1"}],
        "recommendations": [{"text": "r1"}],
        "confidence": 0.7,
    }

    result = await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    assert result.score == pytest.approx(0.7)
    assert any("inconsistent" in r.lower() for r in result.reasons)


@pytest.mark.asyncio
async def test_pricing_no_scenarios():
    pricing = {
        "overall_risk_score": 0.5,
        "premium_band": "Medium",
        "scenarios": [],
        "recommendations": [{"text": "r1"}],
        "confidence": 0.7,
    }

    result = await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    assert result.score == pytest.approx(0.8)
    assert any("No risk scenarios" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_pricing_no_recommendations():
    pricing = {
        "overall_risk_score": 0.5,
        "premium_band": "Medium",
        "scenarios": [{"name": "s1"}],
        "recommendations": [],
        "confidence": 0.7,
    }

    result = await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    assert result.score == pytest.approx(0.8)
    assert any("No recommendations" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_pricing_overconfident():
    pricing = {
        "overall_risk_score": 0.5,
        "premium_band": "Medium",
        "scenarios": [{"name": "s1"}],
        "recommendations": [{"text": "r1"}],
        "confidence": 0.98,
    }

    result = await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    assert result.score == pytest.approx(0.85)
    assert any("Overconfident" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_pricing_logs_to_surrealdb(mock_db_query):
    pricing = {
        "overall_risk_score": 0.5,
        "premium_band": "Medium",
        "scenarios": [{"name": "s1"}],
        "recommendations": [{"text": "r1"}],
        "confidence": 0.7,
    }

    await validate_pricing(pricing, 0.5, 0.5, 0.3, "sess-1")
    mock_db_query.query.assert_called_once()


# --- auto_evaluate_session ---


@pytest.mark.asyncio
async def test_auto_evaluate_complete_session():
    assessment = {
        "legal_analysis": {"exposure": 0.5},
        "technical_analysis": {"risk": 0.6},
        "mitigation_analysis": {"mitigations": []},
        "risk_price": {
            "confidence": 0.8,
            "scenarios": [{"name": "s1"}],
            "recommendations": [{"text": "r1"}],
            "data_gaps": ["gap1"],
        },
    }

    score = await auto_evaluate_session("sess-1", assessment)
    # has_legal=1 + has_technical=1 + has_mitigation=1 + has_pricing=1
    # + confidence=0.8 + has_scenarios=1 + has_recommendations=1 + has_data_gaps=0.8
    # = 7.6 / 8 = 0.95
    assert score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_auto_evaluate_empty_session():
    assessment = {}

    score = await auto_evaluate_session("sess-1", assessment)
    # has_legal=0 + has_technical=0 + has_mitigation=0 + has_pricing=0
    # + confidence=0 + has_scenarios=0 + has_recommendations=0 + has_data_gaps=1
    # = 1.0 / 8 = 0.125
    assert score == pytest.approx(0.125)


@pytest.mark.asyncio
async def test_auto_evaluate_partial_session():
    assessment = {
        "legal_analysis": {"exposure": 0.5},
        "risk_price": {
            "confidence": 0.6,
            "scenarios": [],
            "recommendations": [{"text": "r1"}],
        },
    }

    score = await auto_evaluate_session("sess-1", assessment)
    # has_legal=1 + has_technical=0 + has_mitigation=0 + has_pricing=1
    # + confidence=0.6 + has_scenarios=0 + has_recommendations=1 + has_data_gaps=1
    # = 4.6 / 8 = 0.575
    assert score == pytest.approx(0.575)


@pytest.mark.asyncio
async def test_auto_evaluate_logs_to_surrealdb(mock_db_query):
    await auto_evaluate_session("sess-1", {})
    mock_db_query.query.assert_called_once()
    call_args = mock_db_query.query.call_args
    assert "evaluator = 'auto'" in call_args[0][0]


def test_quality_threshold_value():
    assert QUALITY_THRESHOLD == 0.5
