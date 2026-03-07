"""Tests for prompt version manager (SPEC-09)."""
import pytest
from unittest.mock import AsyncMock, patch

from app.prompts.manager import (
    DEFAULT_PROMPTS,
    seed_default_prompts,
    get_active_prompt,
    create_prompt_version,
    rollback_prompt,
    get_prompt_history,
    record_prompt_performance,
)


@pytest.fixture
def mock_db():
    with patch("app.prompts.manager.db") as m:
        m.query = AsyncMock()
        yield m


class TestDefaultPrompts:
    def test_all_five_agents_have_defaults(self):
        assert set(DEFAULT_PROMPTS.keys()) == {
            "intake", "legal", "technical", "mitigation", "pricing",
        }

    def test_default_prompts_are_nonempty_strings(self):
        for agent, template in DEFAULT_PROMPTS.items():
            assert isinstance(template, str), f"{agent} template is not a string"
            assert len(template) > 100, f"{agent} template is too short"


class TestSeedDefaultPrompts:
    @pytest.mark.asyncio
    async def test_seeds_all_five_agents(self, mock_db):
        # No existing records
        mock_db.query.return_value = [{"result": []}]

        await seed_default_prompts()

        # 5 SELECT checks + 5 CREATEs = 10 calls
        assert mock_db.query.call_count == 10

    @pytest.mark.asyncio
    async def test_idempotent_skips_existing(self, mock_db):
        # All agents already exist
        mock_db.query.return_value = [{"result": [{"agent": "x", "version": 1}]}]

        await seed_default_prompts()

        # Only 5 SELECT checks, no CREATEs
        assert mock_db.query.call_count == 5

    @pytest.mark.asyncio
    async def test_seeds_only_missing_agents(self, mock_db):
        call_count = 0

        async def side_effect(sql, params=None):
            nonlocal call_count
            call_count += 1
            if "SELECT" in sql:
                # First agent exists, rest don't
                if params and params.get("agent") == "intake":
                    return [{"result": [{"agent": "intake", "version": 1}]}]
                return [{"result": []}]
            return [{"result": []}]

        mock_db.query = AsyncMock(side_effect=side_effect)

        await seed_default_prompts()

        # 5 SELECTs + 4 CREATEs (intake skipped) = 9
        assert call_count == 9


class TestGetActivePrompt:
    @pytest.mark.asyncio
    async def test_returns_active_template(self, mock_db):
        mock_db.query.return_value = [{"result": [{"template": "active template v2"}]}]

        result = await get_active_prompt("legal")

        assert result == "active template v2"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_when_no_db_record(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        result = await get_active_prompt("legal")

        assert result == DEFAULT_PROMPTS["legal"]

    @pytest.mark.asyncio
    async def test_returns_empty_for_unknown_agent(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        result = await get_active_prompt("nonexistent")

        assert result == ""


class TestCreatePromptVersion:
    @pytest.mark.asyncio
    async def test_creates_v2_and_deactivates_v1(self, mock_db):
        calls = []

        async def side_effect(sql, params=None):
            calls.append(sql.strip()[:30])
            if "math::max" in sql:
                return [{"result": [{"max_v": 1}]}]
            return [{"result": []}]

        mock_db.query = AsyncMock(side_effect=side_effect)

        version = await create_prompt_version("legal", "new template")

        assert version == 2
        # Should: 1) deactivate old, 2) get max version, 3) create new
        assert mock_db.query.call_count == 3

    @pytest.mark.asyncio
    async def test_creates_v1_for_new_agent(self, mock_db):
        async def side_effect(sql, params=None):
            if "math::max" in sql:
                return [{"result": [{"max_v": None}]}]
            return [{"result": []}]

        mock_db.query = AsyncMock(side_effect=side_effect)

        version = await create_prompt_version("new_agent", "template")

        assert version == 1

    @pytest.mark.asyncio
    async def test_creates_correct_version_number(self, mock_db):
        async def side_effect(sql, params=None):
            if "math::max" in sql:
                return [{"result": [{"max_v": 5}]}]
            return [{"result": []}]

        mock_db.query = AsyncMock(side_effect=side_effect)

        version = await create_prompt_version("legal", "template")

        assert version == 6


class TestRollbackPrompt:
    @pytest.mark.asyncio
    async def test_rollback_to_existing_version(self, mock_db):
        async def side_effect(sql, params=None):
            if "SELECT" in sql:
                return [{"result": [{"agent": "legal", "version": 1}]}]
            return [{"result": []}]

        mock_db.query = AsyncMock(side_effect=side_effect)

        result = await rollback_prompt("legal", 1)

        assert result is True
        # 1 SELECT + 1 deactivate + 1 activate = 3
        assert mock_db.query.call_count == 3

    @pytest.mark.asyncio
    async def test_rollback_to_nonexistent_version_fails(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        result = await rollback_prompt("legal", 99)

        assert result is False
        # Only the SELECT check
        assert mock_db.query.call_count == 1


class TestGetPromptHistory:
    @pytest.mark.asyncio
    async def test_returns_all_versions(self, mock_db):
        versions = [
            {"agent": "legal", "version": 2, "active": True, "performance_score": 0.85},
            {"agent": "legal", "version": 1, "active": False, "performance_score": 0.72},
        ]
        mock_db.query.return_value = [{"result": versions}]

        result = await get_prompt_history("legal")

        assert len(result) == 2
        assert result[0]["version"] == 2
        assert result[1]["version"] == 1

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_history(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        result = await get_prompt_history("nonexistent")

        assert result == []


class TestRecordPromptPerformance:
    @pytest.mark.asyncio
    async def test_updates_active_version_score(self, mock_db):
        async def side_effect(sql, params=None):
            if "SELECT" in sql:
                return [{"result": [{"version": 2}]}]
            return [{"result": []}]

        mock_db.query = AsyncMock(side_effect=side_effect)

        await record_prompt_performance("legal", "session-1", 0.85)

        # 1 SELECT + 1 UPDATE
        assert mock_db.query.call_count == 2

    @pytest.mark.asyncio
    async def test_noop_when_no_active_version(self, mock_db):
        mock_db.query.return_value = [{"result": []}]

        await record_prompt_performance("legal", "session-1", 0.85)

        # Only the SELECT
        assert mock_db.query.call_count == 1
