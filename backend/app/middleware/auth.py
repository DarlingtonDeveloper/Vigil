"""
Layer 1: Authentication & Access Control.
API key auth via X-API-Key header. Keys stored in SurrealDB with scoped permissions.
For the hackathon, support a default "demo" key that bypasses DB lookup.
"""
import hashlib
from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader

from app.db.client import db

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

DEMO_API_KEY = "pg-demo-key-2025"
DEMO_USER = {
    "user_id": "demo-user",
    "scopes": ["read", "write", "admin"],
}

AGENT_SCOPES = {
    "intake": ["surreal:write:deployment_profile", "llm:call"],
    "legal": ["surreal:read:doctrine", "surreal:read:regulation", "llm:call"],
    "technical": ["surreal:read:risk_factor", "llm:call"],
    "mitigation": ["surreal:read:mitigation", "llm:call"],
    "pricing": ["surreal:read:*", "surreal:write:risk_score", "llm:call"],
}


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def get_current_user(api_key: str = Security(api_key_header)) -> dict:
    """Validate API key and return user info with scopes."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if api_key == DEMO_API_KEY:
        return DEMO_USER

    key_hash = hash_key(api_key)
    result = await db.query(
        "SELECT * FROM api_key WHERE key_hash = $hash AND active = true LIMIT 1",
        {"hash": key_hash},
    )

    records = []
    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            records = (
                first["result"]
                if isinstance(first["result"], list)
                else [first["result"]]
            )
        elif isinstance(first, dict):
            records = [first]

    if not records:
        raise HTTPException(status_code=401, detail="Invalid API key")

    record = records[0]
    return {
        "user_id": record["user_id"],
        "scopes": record["scopes"],
    }


def require_scope(scope: str):
    """Dependency that checks for a specific scope."""

    async def check(user: dict = Depends(get_current_user)):
        if "admin" in user["scopes"]:
            return user
        if scope not in user["scopes"]:
            raise HTTPException(status_code=403, detail=f"Missing scope: {scope}")
        return user

    return check
