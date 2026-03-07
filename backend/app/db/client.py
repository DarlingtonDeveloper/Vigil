from surrealdb import AsyncSurreal

from app.config import settings


class DatabaseClient:
    """Async SurrealDB client wrapper with knowledge graph query helpers."""

    def __init__(self) -> None:
        self._db = None

    async def connect(self) -> None:
        self._db = AsyncSurreal(settings.surreal_url)
        await self._db.connect()
        await self._db.signin({"username": settings.surreal_user, "password": settings.surreal_pass})
        await self._db.use(settings.surreal_ns, settings.surreal_db)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def connected(self) -> bool:
        return self._db is not None

    async def query(self, sql: str, params: dict | None = None) -> list:
        if not self._db:
            await self.connect()
        result = await self._db.query(sql, vars=params)
        if isinstance(result, dict):
            return [result]
        return result

    async def get_applicable_doctrines(
        self, jurisdictions: list[str], domains: list[str]
    ) -> list[dict]:
        result = await self.query(
            """
            SELECT * FROM legal_doctrine
            WHERE jurisdiction IN $jurisdictions
            AND domain IN $domains
            ORDER BY relevance_score DESC
            """,
            {"jurisdictions": jurisdictions, "domains": domains},
        )
        return _flatten(result)

    async def get_applicable_regulations(self, jurisdictions: list[str]) -> list[dict]:
        result = await self.query(
            """
            SELECT * FROM regulation
            WHERE jurisdiction IN $jurisdictions
            AND status = 'active'
            ORDER BY severity DESC
            """,
            {"jurisdictions": jurisdictions},
        )
        return _flatten(result)


def _flatten(result: list) -> list:
    """Extract result rows from SurrealDB response wrapper."""
    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            return first["result"]
        return result
    return result or []


# Global instance
db = DatabaseClient()
