"""Tests for cost tracking (SPEC-09)."""
import pytest
from unittest.mock import AsyncMock, patch

from app.tracing.cost_tracker import (
    PRICING,
    calculate_cost,
    track_llm_cost,
    get_session_cost,
    get_daily_cost,
    get_average_cost_per_run,
)


MODEL = "claude-sonnet-4-20250514"


class TestCalculateCost:
    def test_zero_tokens(self):
        assert calculate_cost(MODEL, 0, 0) == 0.0

    def test_input_only(self):
        cost = calculate_cost(MODEL, 1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_output_only(self):
        cost = calculate_cost(MODEL, 0, 1_000_000)
        assert cost == pytest.approx(15.0)

    def test_mixed_tokens(self):
        cost = calculate_cost(MODEL, 1000, 500)
        expected = 1000 * (3.0 / 1_000_000) + 500 * (15.0 / 1_000_000)
        assert cost == pytest.approx(expected)

    def test_unknown_model_uses_default(self):
        cost = calculate_cost("unknown-model", 1_000_000, 0)
        assert cost == pytest.approx(3.0)

    def test_realistic_call(self):
        # Typical call: 2000 input, 800 output
        cost = calculate_cost(MODEL, 2000, 800)
        expected = 2000 * 3.0 / 1e6 + 800 * 15.0 / 1e6
        assert cost == pytest.approx(expected)
        assert cost == pytest.approx(0.018)


@pytest.fixture
def mock_db():
    with patch("app.tracing.cost_tracker.db") as m:
        m.query = AsyncMock(return_value=[{"result": []}])
        yield m


@pytest.fixture
def mock_audit():
    with patch("app.tracing.cost_tracker.log_audit", new_callable=AsyncMock) as m:
        yield m


class TestTrackLLMCost:
    @pytest.mark.asyncio
    async def test_writes_audit_log(self, mock_db, mock_audit):
        await track_llm_cost(
            session_id="sess-1",
            agent="legal",
            model=MODEL,
            input_tokens=1000,
            output_tokens=500,
        )

        mock_audit.assert_called_once()
        call_kwargs = mock_audit.call_args.kwargs
        assert call_kwargs["session_id"] == "sess-1"
        assert call_kwargs["agent"] == "legal"
        assert call_kwargs["action"] == "llm_call"
        assert call_kwargs["token_usage"]["input"] == 1000
        assert call_kwargs["token_usage"]["output"] == 500
        assert call_kwargs["token_usage"]["total"] == 1500
        assert call_kwargs["cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_updates_cost_summary(self, mock_db, mock_audit):
        await track_llm_cost(
            session_id="sess-1",
            agent="legal",
            model=MODEL,
            input_tokens=1000,
            output_tokens=500,
        )

        # Should call db.query for the UPSERT cost_summary
        assert mock_db.query.call_count == 1
        call_sql = mock_db.query.call_args[0][0]
        assert "UPSERT cost_summary" in call_sql

    @pytest.mark.asyncio
    async def test_cost_matches_calculate_cost(self, mock_db, mock_audit):
        await track_llm_cost(
            session_id="sess-1",
            agent="legal",
            model=MODEL,
            input_tokens=2000,
            output_tokens=800,
        )

        expected_cost = calculate_cost(MODEL, 2000, 800)
        call_kwargs = mock_audit.call_args.kwargs
        assert call_kwargs["cost_usd"] == pytest.approx(expected_cost)


class TestGetSessionCost:
    @pytest.mark.asyncio
    async def test_returns_session_totals(self, mock_db):
        mock_db.query.return_value = [{"result": [
            {"total_cost": 0.045, "total_tokens": 4500, "llm_calls": 3}
        ]}]

        result = await get_session_cost("sess-1")

        assert result["total_cost"] == 0.045
        assert result["total_tokens"] == 4500
        assert result["llm_calls"] == 3

    @pytest.mark.asyncio
    async def test_returns_zeros_for_unknown_session(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        result = await get_session_cost("nonexistent")

        assert result == {"total_cost": 0, "total_tokens": 0, "llm_calls": 0}


class TestGetDailyCost:
    @pytest.mark.asyncio
    async def test_returns_daily_summaries(self, mock_db):
        days = [
            {"period": "2025-03-07", "total_cost_usd": 1.50, "total_tokens": 50000},
            {"period": "2025-03-06", "total_cost_usd": 2.30, "total_tokens": 80000},
        ]
        mock_db.query.return_value = [{"result": days}]

        result = await get_daily_cost("demo-user", days=7)

        assert len(result) == 2
        assert result[0]["period"] == "2025-03-07"

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_data(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        result = await get_daily_cost("new-user")

        assert result == []


class TestGetAverageCostPerRun:
    @pytest.mark.asyncio
    async def test_returns_average(self, mock_db):
        mock_db.query.return_value = [{"result": [{"avg_cost_per_run": 0.025}]}]

        result = await get_average_cost_per_run("demo-user")

        assert result == pytest.approx(0.025)

    @pytest.mark.asyncio
    async def test_returns_zero_for_no_data(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        result = await get_average_cost_per_run("new-user")

        assert result == 0
