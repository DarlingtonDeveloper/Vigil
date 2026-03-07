"""
SurrealDB async client singleton.

Provides a module-level `db` instance used throughout the backend.
Call `db.connect()` at app startup and `db.close()` at shutdown.
"""
from surrealdb import AsyncSurreal

from app.config import settings


class SurrealClient:
    """Thin wrapper around AsyncSurreal for lifecycle management."""

    def __init__(self):
        self._client: AsyncSurreal | None = None

    async def connect(self):
        """Connect, authenticate, and select namespace/database."""
        self._client = AsyncSurreal(settings.surreal_url)
        await self._client.connect()
        await self._client.signin(
            {"username": settings.surreal_user, "password": settings.surreal_pass}
        )
        await self._client.use(settings.surreal_ns, settings.surreal_db)

    async def close(self):
        """Close the connection."""
        if self._client:
            await self._client.close()
            self._client = None

    async def query(self, sql: str, params: dict | None = None):
        """Execute a SurrealQL query with optional parameters."""
        if not self._client:
            raise RuntimeError("SurrealDB client not connected. Call connect() first.")
        return await self._client.query(sql, params)

    @property
    def connected(self) -> bool:
        return self._client is not None


db = SurrealClient()
