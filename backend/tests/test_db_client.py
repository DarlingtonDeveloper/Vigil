"""Unit tests for SurrealClient."""
from unittest.mock import patch, mock_open


class TestSurrealClientInit:
    """Test client initialization from settings."""

    def test_defaults_from_settings(self):
        from app.db.client import SurrealClient
        c = SurrealClient()
        assert c._url == "ws://localhost:8000/rpc"
        assert c._namespace == "faultline"
        assert c._database == "faultline"
        assert c._username == "root"
        assert c._password == "root"
        assert c._conn is None

    def test_module_singleton(self):
        from app.db.client import db
        from app.db.client import SurrealClient
        assert isinstance(db, SurrealClient)


class TestSurrealClientConnect:
    """Test connect/disconnect lifecycle."""

    async def test_connect(self, fresh_db_client, mock_conn):
        with patch("app.db.client.AsyncSurreal", return_value=mock_conn):
            await fresh_db_client.connect()
            mock_conn.connect.assert_awaited_once()
            mock_conn.signin.assert_awaited_once_with(
                {"username": "root", "password": "root"}
            )
            mock_conn.use.assert_awaited_once_with("faultline", "faultline")
            assert fresh_db_client._conn is mock_conn

    async def test_disconnect(self, db_client, mock_conn):
        await db_client.disconnect()
        mock_conn.close.assert_awaited_once()
        assert db_client._conn is None

    async def test_disconnect_when_not_connected(self, fresh_db_client):
        await fresh_db_client.disconnect()
        assert fresh_db_client._conn is None


class TestSurrealClientCRUD:
    """Test basic CRUD delegation."""

    async def test_query_without_params(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": [{"count": 5}]}]
        result = await db_client.query("SELECT count() FROM foo GROUP ALL")
        mock_conn.query.assert_awaited_once_with("SELECT count() FROM foo GROUP ALL")
        assert result == [{"result": [{"count": 5}]}]

    async def test_query_with_params(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": [{"name": "bar"}]}]
        await db_client.query("SELECT * FROM foo WHERE name = $n", {"n": "bar"})
        mock_conn.query.assert_awaited_once_with(
            "SELECT * FROM foo WHERE name = $n", {"n": "bar"}
        )

    async def test_create(self, db_client, mock_conn):
        mock_conn.create.return_value = {"id": "foo:abc", "name": "test"}
        result = await db_client.create("foo", {"name": "test"})
        mock_conn.create.assert_awaited_once_with("foo", {"name": "test"})
        assert result["id"] == "foo:abc"

    async def test_select(self, db_client, mock_conn):
        mock_conn.select.return_value = [{"id": "foo:1"}, {"id": "foo:2"}]
        result = await db_client.select("foo")
        mock_conn.select.assert_awaited_once_with("foo")
        assert len(result) == 2

    async def test_update(self, db_client, mock_conn):
        mock_conn.update.return_value = {"id": "foo:1", "name": "updated"}
        result = await db_client.update("foo:1", {"name": "updated"})
        mock_conn.update.assert_awaited_once_with("foo:1", {"name": "updated"})
        assert result["name"] == "updated"

    async def test_delete(self, db_client, mock_conn):
        await db_client.delete("foo:1")
        mock_conn.delete.assert_awaited_once_with("foo:1")


class TestSurrealClientSchema:
    """Test schema execution."""

    async def test_execute_schema_default_path(self, db_client, mock_conn):
        schema_content = "DEFINE TABLE foo SCHEMAFULL;"
        with patch("builtins.open", mock_open(read_data=schema_content)):
            await db_client.execute_schema()
            mock_conn.query.assert_awaited_once_with(schema_content)

    async def test_execute_schema_custom_path(self, db_client, mock_conn):
        schema_content = "DEFINE TABLE bar SCHEMAFULL;"
        with patch("builtins.open", mock_open(read_data=schema_content)):
            await db_client.execute_schema("/tmp/custom.surql")
            mock_conn.query.assert_awaited_once_with(schema_content)


class TestSurrealClientHelpers:
    """Test FaultLine-specific helper methods."""

    async def test_get_applicable_doctrines(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": [{"name": "apparent_authority"}]}]
        await db_client.get_applicable_doctrines(["UK"], ["contract_law"])
        call_args = mock_conn.query.call_args
        assert "$jurisdictions" in call_args[0][0]
        assert "$domains" in call_args[0][0]
        assert call_args[0][1] == {"jurisdictions": ["UK"], "domains": ["contract_law"]}

    async def test_get_applicable_regulations(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": [{"short_name": "GDPR"}]}]
        await db_client.get_applicable_regulations(["EU/UK"])
        call_args = mock_conn.query.call_args
        assert "status IN ['in_force', 'partial']" in call_args[0][0]
        assert call_args[0][1] == {"jurisdictions": ["EU/UK"]}

    async def test_get_risk_factors_by_category(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": [{"name": "autonomy_level", "weight": 0.9}]}]
        await db_client.get_risk_factors_by_category("technical")
        call_args = mock_conn.query.call_args
        assert "ORDER BY weight DESC" in call_args[0][0]
        assert call_args[0][1] == {"cat": "technical"}

    async def test_get_mitigations_for_risk(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": [{"mitigation_name": "qualified_hitl", "reduction": 0.5}]}]
        await db_client.get_mitigations_for_risk("hallucination_risk")
        call_args = mock_conn.query.call_args
        assert "FROM mitigates" in call_args[0][0]
        assert call_args[0][1] == {"rf_name": "hallucination_risk"}

    async def test_get_doctrine_relationships(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": [{"related_doctrine": "vicarious_liability"}]}]
        await db_client.get_doctrine_relationships("apparent_authority")
        call_args = mock_conn.query.call_args
        assert "FROM doctrine_relates" in call_args[0][0]
        assert call_args[0][1] == {"name": "apparent_authority"}

    async def test_get_knowledge_graph_full(self, db_client, mock_conn):
        mock_conn.query.return_value = [{"result": []}]
        result = await db_client.get_knowledge_graph_full()
        assert set(result.keys()) == {
            "doctrines", "regulations", "risk_factors",
            "mitigations", "doctrine_edges", "mitigation_edges",
        }
        assert mock_conn.query.await_count == 6
