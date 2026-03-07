import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_conn():
    """A mock AsyncSurreal connection."""
    conn = AsyncMock()
    conn.connect = AsyncMock()
    conn.signin = AsyncMock()
    conn.use = AsyncMock()
    conn.close = AsyncMock()
    conn.query = AsyncMock(return_value=[])
    conn.create = AsyncMock(return_value={})
    conn.select = AsyncMock(return_value=[])
    conn.update = AsyncMock(return_value={})
    conn.delete = AsyncMock()
    return conn


@pytest.fixture
def db_client(mock_conn):
    """A SurrealClient with a mocked connection (already 'connected')."""
    with patch("app.db.client.AsyncSurreal", return_value=mock_conn):
        from app.db.client import SurrealClient
        c = SurrealClient()
        c._conn = mock_conn
        return c


@pytest.fixture
def fresh_db_client():
    """A SurrealClient with no connection (for testing connect/disconnect)."""
    from app.db.client import SurrealClient
    return SurrealClient()
