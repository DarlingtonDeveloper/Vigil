"""
SurrealDB async client wrapper.

Provides a module-level `db` instance used throughout the backend.
Call `connect()` at startup and `disconnect()` at shutdown.
"""
from surrealdb import AsyncSurreal

from app.config import settings


class SurrealClient:
    """Thin wrapper around the SurrealDB async WebSocket connection."""

    def __init__(self):
        self._conn = None

    async def connect(self):
        self._conn = AsyncSurreal(settings.surreal_url)
        await self._conn.connect(settings.surreal_url)
        await self._conn.signin({"username": settings.surreal_user, "password": settings.surreal_pass})
        await self._conn.use(settings.surreal_ns, settings.surreal_db)

    async def disconnect(self):
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def query(self, sql: str, vars: dict | None = None):
        if self._conn is None:
            raise RuntimeError("SurrealDB client not connected. Call connect() first.")
        return await self._conn.query(sql, vars)


db = SurrealClient()
