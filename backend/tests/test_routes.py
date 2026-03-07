"""Tests for SPEC-08: FastAPI endpoints."""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.middleware.auth import DEMO_API_KEY

client = TestClient(app)

HEADERS = {"X-API-Key": DEMO_API_KEY}

VALID_DESCRIPTION = (
    "An AI agent that autonomously reviews legal contracts, extracts key clauses, "
    "identifies risk factors, and generates summary reports for the legal team."
)


# ---------------------------------------------------------------------------
# POST /api/assess
# ---------------------------------------------------------------------------


class TestAssessEndpoint:
    """POST /api/assess — triggers full LangGraph pipeline."""

    def test_missing_api_key_returns_401(self):
        resp = client.post("/api/assess", json={
            "description": VALID_DESCRIPTION,
        })
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self):
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[{"result": []}])
        with patch("app.middleware.auth.db", mock_db):
            resp = client.post(
                "/api/assess",
                json={"description": VALID_DESCRIPTION},
                headers={"X-API-Key": "bad-key"},
            )
        assert resp.status_code == 401

    @patch("app.middleware.rate_limit.db")
    def test_short_description_returns_422(self, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        resp = client.post(
            "/api/assess",
            json={"description": "Too short"},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    @patch("app.middleware.rate_limit.db")
    def test_missing_description_returns_422(self, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        resp = client.post("/api/assess", json={}, headers=HEADERS)
        assert resp.status_code == 422

    @patch("app.middleware.rate_limit.db")
    @patch("app.routes.assess.run_assessment")
    def test_successful_assessment(self, mock_run, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        mock_run.return_value = {
            "session_id": "sess-123",
            "risk_price": {"overall_risk_score": 0.65, "premium_band": "medium"},
            "deployment_profile": {"agent_description": "test"},
            "legal_analysis": {"legal_exposure_score": 0.5},
            "technical_analysis": {"technical_risk_score": 0.6},
            "mitigation_analysis": {"overall_mitigation_score": 0.4},
        }

        resp = client.post(
            "/api/assess",
            json={
                "description": VALID_DESCRIPTION,
                "jurisdictions": ["UK"],
                "sector": "legal",
            },
            headers=HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-123"
        assert data["status"] == "completed"
        assert data["risk_price"]["overall_risk_score"] == 0.65
        assert data["deployment_profile"] is not None
        assert data["legal_analysis"] is not None
        assert data["technical_analysis"] is not None
        assert data["mitigation_analysis"] is not None

    @patch("app.middleware.rate_limit.db")
    @patch("app.routes.assess.run_assessment")
    def test_assessment_with_default_jurisdictions(self, mock_run, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        mock_run.return_value = {"session_id": "sess-456"}

        resp = client.post(
            "/api/assess",
            json={"description": VALID_DESCRIPTION},
            headers=HEADERS,
        )

        assert resp.status_code == 200
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["jurisdictions"] == ["UK"]

    @patch("app.middleware.rate_limit.db")
    @patch("app.routes.assess.run_assessment")
    def test_assessment_pipeline_failure_returns_500(self, mock_run, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        mock_run.side_effect = RuntimeError("LLM provider unavailable")

        resp = client.post(
            "/api/assess",
            json={"description": VALID_DESCRIPTION},
            headers=HEADERS,
        )

        assert resp.status_code == 500
        assert "Assessment failed" in resp.json()["detail"]

    def test_rate_limit_exceeded_returns_429(self):
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[{
            "result": [{
                "user_id": "demo-user",
                "request_count": 999,
                "token_budget_used": 0,
                "token_budget_max": 100000,
                "window_start": "2026-01-01T00:00:00+00:00",
            }]
        }])
        with patch("app.middleware.rate_limit.db", mock_db):
            resp = client.post(
                "/api/assess",
                json={"description": VALID_DESCRIPTION},
                headers=HEADERS,
            )
        assert resp.status_code == 429

    @patch("app.middleware.rate_limit.db")
    @patch("app.routes.assess.run_assessment")
    def test_invalid_jurisdiction_filtered(self, mock_run, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        mock_run.return_value = {"session_id": "sess-789"}

        resp = client.post(
            "/api/assess",
            json={
                "description": VALID_DESCRIPTION,
                "jurisdictions": ["INVALID", "UK"],
            },
            headers=HEADERS,
        )

        assert resp.status_code == 200
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs["jurisdictions"] == ["UK"]


# ---------------------------------------------------------------------------
# POST /api/scenario
# ---------------------------------------------------------------------------


class TestScenarioEndpoint:
    """POST /api/scenario — run risk scenario on existing assessment."""

    def test_missing_api_key_returns_401(self):
        resp = client.post("/api/scenario", json={
            "session_id": "sess-123",
            "scenario_type": "hallucination",
        })
        assert resp.status_code == 401

    @patch("app.middleware.rate_limit.db")
    def test_invalid_scenario_type_returns_422(self, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        resp = client.post(
            "/api/scenario",
            json={"session_id": "s1", "scenario_type": "invalid_type"},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    @patch("app.middleware.rate_limit.db")
    @patch("app.routes.scenario.db")
    def test_assessment_not_found_returns_400(self, mock_route_db, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])
        mock_route_db.query = AsyncMock(return_value=[])

        resp = client.post(
            "/api/scenario",
            json={"session_id": "nonexistent", "scenario_type": "hallucination"},
            headers=HEADERS,
        )

        assert resp.status_code == 400
        assert "Assessment not found" in resp.json()["detail"]

    @patch("app.middleware.rate_limit.db")
    @patch("app.routes.scenario.db")
    def test_successful_scenario_query(self, mock_route_db, mock_rl_db):
        mock_rl_db.query = AsyncMock(return_value=[])

        call_count = 0

        async def side_effect(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # profile and risk_score queries
                return [{"result": [{"session_id": "sess-1"}]}]
            else:
                # scenario query
                return [{"result": [
                    {"scenario_type": "hallucination", "description": "Agent hallucinates"},
                ]}]

        mock_route_db.query = AsyncMock(side_effect=side_effect)

        resp = client.post(
            "/api/scenario",
            json={"session_id": "sess-1", "scenario_type": "hallucination"},
            headers=HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-1"
        assert data["scenario_type"] == "hallucination"
        assert isinstance(data["results"], list)

    @patch("app.middleware.rate_limit.db")
    @patch("app.routes.scenario.db")
    def test_scenario_with_direct_records(self, mock_route_db, mock_rl_db):
        """Test scenario parsing when DB returns records directly (no result wrapper)."""
        mock_rl_db.query = AsyncMock(return_value=[])

        call_count = 0

        async def side_effect(sql, params=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return [{"result": [{"session_id": "sess-1"}]}]
            else:
                return [
                    {"scenario_type": "data_breach", "description": "Data leaked"},
                ]

        mock_route_db.query = AsyncMock(side_effect=side_effect)

        resp = client.post(
            "/api/scenario",
            json={"session_id": "sess-1", "scenario_type": "data_breach"},
            headers=HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["scenario_type"] == "data_breach"


# ---------------------------------------------------------------------------
# GET /api/knowledge/*
# ---------------------------------------------------------------------------


class TestKnowledgeEndpoints:
    """Knowledge graph query endpoints."""

    def test_full_graph_no_auth_returns_401(self):
        resp = client.get("/api/knowledge/full")
        assert resp.status_code == 401

    @patch("app.routes.knowledge.db")
    def test_full_knowledge_graph(self, mock_db):
        mock_db.get_knowledge_graph_full = AsyncMock(return_value={
            "doctrines": [{"name": "apparent_authority"}],
            "regulations": [{"short_name": "GDPR"}],
            "risk_factors": [{"name": "hallucination_risk"}],
            "mitigations": [{"name": "human_oversight"}],
            "doctrine_edges": [],
            "mitigation_edges": [],
        })

        resp = client.get("/api/knowledge/full", headers=HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert "doctrines" in data
        assert "regulations" in data
        assert "risk_factors" in data
        assert "mitigations" in data
        assert data["doctrines"][0]["name"] == "apparent_authority"

    @patch("app.routes.knowledge.get_knowledge_stats")
    def test_knowledge_stats(self, mock_stats):
        mock_stats.return_value = {
            "doctrines": 15,
            "regulations": 8,
            "risk_factors": 22,
            "mitigations": 18,
            "mitigation_edges": 45,
        }

        resp = client.get("/api/knowledge/stats", headers=HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert data["doctrines"] == 15
        assert data["regulations"] == 8

    @patch("app.routes.knowledge.db")
    def test_doctrines_default_jurisdiction(self, mock_db):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[
            {"name": "vicarious_liability", "jurisdiction": "UK"},
        ])

        resp = client.get("/api/knowledge/doctrines", headers=HEADERS)

        assert resp.status_code == 200
        mock_db.get_applicable_doctrines.assert_called_once_with(
            ["UK", "global"],
            [
                "contract_law", "tort_law", "data_protection",
                "regulatory_compliance", "intellectual_property",
                "employment_law",
            ],
        )

    @patch("app.routes.knowledge.db")
    def test_doctrines_eu_jurisdiction(self, mock_db):
        mock_db.get_applicable_doctrines = AsyncMock(return_value=[])

        resp = client.get(
            "/api/knowledge/doctrines?jurisdiction=EU",
            headers=HEADERS,
        )

        assert resp.status_code == 200
        mock_db.get_applicable_doctrines.assert_called_once_with(
            ["EU", "global"],
            [
                "contract_law", "tort_law", "data_protection",
                "regulatory_compliance", "intellectual_property",
                "employment_law",
            ],
        )

    @patch("app.routes.knowledge.db")
    def test_regulations_default(self, mock_db):
        mock_db.get_applicable_regulations = AsyncMock(return_value=[
            {"short_name": "GDPR", "jurisdiction": "UK"},
        ])

        resp = client.get("/api/knowledge/regulations", headers=HEADERS)

        assert resp.status_code == 200
        mock_db.get_applicable_regulations.assert_called_once_with(["UK", "global"])

    @patch("app.routes.knowledge.db")
    def test_regulations_eu(self, mock_db):
        mock_db.get_applicable_regulations = AsyncMock(return_value=[
            {"short_name": "AI Act", "jurisdiction": "EU"},
        ])

        resp = client.get(
            "/api/knowledge/regulations?jurisdiction=EU",
            headers=HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["short_name"] == "AI Act"

    @patch("app.routes.knowledge.db")
    def test_risk_factors_all(self, mock_db):
        mock_db.query = AsyncMock(return_value=[
            {"name": "hallucination_risk", "weight": 0.9},
            {"name": "data_breach", "weight": 0.8},
        ])

        resp = client.get("/api/knowledge/risk-factors", headers=HEADERS)

        assert resp.status_code == 200
        mock_db.query.assert_called_once()

    @patch("app.routes.knowledge.db")
    def test_risk_factors_by_category(self, mock_db):
        mock_db.get_risk_factors_by_category = AsyncMock(return_value=[
            {"name": "hallucination_risk", "category": "operational"},
        ])

        resp = client.get(
            "/api/knowledge/risk-factors?category=operational",
            headers=HEADERS,
        )

        assert resp.status_code == 200
        mock_db.get_risk_factors_by_category.assert_called_once_with("operational")

    @patch("app.routes.knowledge.db")
    def test_mitigations_for_risk(self, mock_db):
        mock_db.get_mitigations_for_risk = AsyncMock(return_value=[
            {"mitigation_name": "human_oversight", "reduction": 0.3},
        ])

        resp = client.get(
            "/api/knowledge/mitigations/hallucination_risk",
            headers=HEADERS,
        )

        assert resp.status_code == 200
        mock_db.get_mitigations_for_risk.assert_called_once_with("hallucination_risk")

    @patch("app.routes.knowledge.db")
    def test_doctrine_relationships(self, mock_db):
        mock_db.get_doctrine_relationships = AsyncMock(return_value=[
            {"related_doctrine": "vicarious_liability", "relationship": "extends"},
        ])

        resp = client.get(
            "/api/knowledge/doctrine/apparent_authority/relationships",
            headers=HEADERS,
        )

        assert resp.status_code == 200
        mock_db.get_doctrine_relationships.assert_called_once_with("apparent_authority")

    @patch("app.routes.knowledge.db")
    def test_audit_trail(self, mock_db):
        mock_db.query = AsyncMock(return_value=[{"result": [
            {"agent": "intake", "action": "start", "timestamp": "2026-01-01T00:00:00Z"},
            {"agent": "intake", "action": "complete", "timestamp": "2026-01-01T00:00:01Z"},
        ]}])

        resp = client.get(
            "/api/knowledge/audit/sess-123",
            headers=HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["agent"] == "intake"

    @patch("app.routes.knowledge.db")
    def test_audit_trail_empty(self, mock_db):
        mock_db.query = AsyncMock(return_value=[])

        resp = client.get(
            "/api/knowledge/audit/nonexistent",
            headers=HEADERS,
        )

        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# POST/GET /api/feedback
# ---------------------------------------------------------------------------


class TestFeedbackEndpoints:
    """Feedback submission and retrieval."""

    def test_feedback_no_auth_returns_401(self):
        resp = client.post("/api/feedback", json={
            "session_id": "s1",
            "feedback_type": "thumbs_up",
        })
        assert resp.status_code == 401

    @patch("app.routes.feedback.db")
    def test_submit_thumbs_up(self, mock_db):
        mock_db.query = AsyncMock(return_value=[])

        resp = client.post(
            "/api/feedback",
            json={"session_id": "sess-1", "feedback_type": "thumbs_up"},
            headers=HEADERS,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["session_id"] == "sess-1"

        # Verify default score of 1.0 for thumbs_up
        call_args = mock_db.query.call_args
        assert call_args.args[1]["score"] == 1.0

    @patch("app.routes.feedback.db")
    def test_submit_thumbs_down(self, mock_db):
        mock_db.query = AsyncMock(return_value=[])

        resp = client.post(
            "/api/feedback",
            json={"session_id": "sess-1", "feedback_type": "thumbs_down"},
            headers=HEADERS,
        )

        assert resp.status_code == 200
        call_args = mock_db.query.call_args
        assert call_args.args[1]["score"] == 0.0

    @patch("app.routes.feedback.db")
    def test_submit_feedback_with_explicit_score(self, mock_db):
        mock_db.query = AsyncMock(return_value=[])

        resp = client.post(
            "/api/feedback",
            json={
                "session_id": "sess-1",
                "feedback_type": "override",
                "score": 0.75,
                "detail": "Good but could improve scenario coverage",
            },
            headers=HEADERS,
        )

        assert resp.status_code == 200
        call_args = mock_db.query.call_args
        assert call_args.args[1]["score"] == 0.75
        assert "could improve" in call_args.args[1]["detail"]

    def test_invalid_feedback_type_returns_422(self):
        resp = client.post(
            "/api/feedback",
            json={"session_id": "s1", "feedback_type": "invalid"},
            headers=HEADERS,
        )
        assert resp.status_code == 422

    @patch("app.routes.feedback.db")
    def test_get_feedback(self, mock_db):
        mock_db.query = AsyncMock(return_value=[{"result": [
            {"session_id": "sess-1", "feedback_type": "thumbs_up", "score": 1.0},
        ]}])

        resp = client.get("/api/feedback/sess-1", headers=HEADERS)

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["feedback_type"] == "thumbs_up"

    @patch("app.routes.feedback.db")
    def test_get_feedback_empty(self, mock_db):
        mock_db.query = AsyncMock(return_value=[])

        resp = client.get("/api/feedback/nonexistent", headers=HEADERS)

        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_feedback_no_auth_returns_401(self):
        resp = client.get("/api/feedback/sess-1")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Auth middleware applied to all endpoints
# ---------------------------------------------------------------------------


class TestAuthMiddlewareOnAllEndpoints:
    """Verify auth middleware is enforced on every endpoint."""

    @pytest.mark.parametrize("method,path", [
        ("post", "/api/assess"),
        ("post", "/api/scenario"),
        ("get", "/api/knowledge/full"),
        ("get", "/api/knowledge/stats"),
        ("get", "/api/knowledge/doctrines"),
        ("get", "/api/knowledge/regulations"),
        ("get", "/api/knowledge/risk-factors"),
        ("get", "/api/knowledge/mitigations/test"),
        ("get", "/api/knowledge/doctrine/test/relationships"),
        ("get", "/api/knowledge/audit/test"),
        ("post", "/api/feedback"),
        ("get", "/api/feedback/test"),
    ])
    def test_endpoint_requires_auth(self, method, path):
        handler = getattr(client, method)
        if method == "post":
            resp = handler(path, json={})
        else:
            resp = handler(path)
        # Should get 401 (missing key) or 422 (validation error after auth)
        # but never 200 without auth
        assert resp.status_code in (401, 403), (
            f"{method.upper()} {path} returned {resp.status_code} without auth"
        )


# ---------------------------------------------------------------------------
# Rate limiting on assess/scenario
# ---------------------------------------------------------------------------


class TestRateLimitingOnEndpoints:
    """Verify rate limiting middleware on assessment and scenario endpoints."""

    @pytest.mark.parametrize("path,payload", [
        ("/api/assess", {"description": VALID_DESCRIPTION}),
        ("/api/scenario", {"session_id": "s1", "scenario_type": "hallucination"}),
    ])
    def test_rate_limited_endpoint(self, path, payload):
        """Rate-limited endpoints use check_rate_limit (which chains get_current_user)."""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[{
            "result": [{
                "user_id": "demo-user",
                "request_count": 999,
                "token_budget_used": 0,
                "token_budget_max": 100000,
                "window_start": "2026-01-01T00:00:00+00:00",
            }]
        }])
        with patch("app.middleware.rate_limit.db", mock_db):
            resp = client.post(path, json=payload, headers=HEADERS)
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# OpenAPI schema includes all routes
# ---------------------------------------------------------------------------


class TestOpenAPISchema:
    """Verify all routes are registered in the OpenAPI schema."""

    def test_all_routes_in_openapi(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        paths = resp.json()["paths"]

        expected = [
            "/api/assess",
            "/api/scenario",
            "/api/knowledge/full",
            "/api/knowledge/stats",
            "/api/knowledge/doctrines",
            "/api/knowledge/regulations",
            "/api/knowledge/risk-factors",
            "/api/knowledge/mitigations/{risk_factor}",
            "/api/knowledge/doctrine/{doctrine_name}/relationships",
            "/api/knowledge/audit/{session_id}",
            "/api/feedback",
            "/api/feedback/{session_id}",
            "/health",
        ]

        for path in expected:
            assert path in paths, f"Missing route: {path}"

    def test_tags_on_routes(self):
        resp = client.get("/openapi.json")
        paths = resp.json()["paths"]

        # Collect all tags used across all operations
        all_tags = set()
        for path_data in paths.values():
            for method_data in path_data.values():
                if isinstance(method_data, dict) and "tags" in method_data:
                    all_tags.update(method_data["tags"])

        assert "assessment" in all_tags
        assert "scenarios" in all_tags
        assert "knowledge" in all_tags
        assert "feedback" in all_tags
