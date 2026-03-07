"""
Layer 2: Input Validation & Data Sanitisation.
Pydantic models for all API inputs. Strips PII, validates formats, blocks prompt injection.
"""
import re
from typing import Optional
from pydantic import BaseModel, field_validator

INJECTION_PATTERNS = [
    r"(?i)(ignore|forget|disregard)\s+(previous|above|all|prior)\s+(instructions?|prompts?|rules?)",
    r"(?i)you\s+are\s+now\s+",
    r"(?i)system\s*:\s*",
    r"(?i)jailbreak",
    r"(?i)DAN\s+mode",
]

PII_PATTERNS = {
    "email": (r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL_REDACTED]"),
    "phone": (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE_REDACTED]"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    "credit_card": (
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "[CC_REDACTED]",
    ),
}


def sanitise_text(text: str) -> str:
    """Strip prompt injection patterns and PII from text."""
    for pattern in INJECTION_PATTERNS:
        text = re.sub(pattern, "", text)
    for name, (pattern, replacement) in PII_PATTERNS.items():
        text = re.sub(pattern, replacement, text)
    return text.strip()


class AssessDeploymentRequest(BaseModel):
    """Request to assess an agentic deployment's risk."""

    description: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    jurisdictions: list[str] = ["UK"]

    @field_validator("description")
    @classmethod
    def sanitise_description(cls, v):
        v = sanitise_text(v)
        if len(v) > 10000:
            raise ValueError("Description too long (max 10000 chars)")
        if len(v) < 50:
            raise ValueError(
                "Description too short — please provide more detail about the agent's capabilities, tools, and data access"
            )
        return v

    @field_validator("jurisdictions")
    @classmethod
    def validate_jurisdictions(cls, v):
        valid = ["UK", "EU", "US", "global"]
        filtered = [j for j in v if j in valid]
        return filtered or ["UK"]


class RunRiskScenarioRequest(BaseModel):
    """Request to simulate a specific risk scenario."""

    session_id: str
    scenario_type: str
    description: Optional[str] = None

    @field_validator("scenario_type")
    @classmethod
    def validate_type(cls, v):
        valid = [
            "hallucination",
            "contract_formation",
            "data_breach",
            "scope_creep",
            "adversarial",
            "regulatory_breach",
        ]
        if v not in valid:
            raise ValueError(f"scenario_type must be one of: {valid}")
        return v


class FeedbackRequest(BaseModel):
    """User feedback on an assessment."""

    session_id: str
    feedback_type: str
    score: Optional[float] = None
    detail: Optional[str] = None

    @field_validator("feedback_type")
    @classmethod
    def validate_type(cls, v):
        valid = ["thumbs_up", "thumbs_down", "override"]
        if v not in valid:
            raise ValueError(f"feedback_type must be one of: {valid}")
        return v

    @field_validator("detail")
    @classmethod
    def sanitise_detail(cls, v):
        if v:
            return sanitise_text(v)[:1000]
        return v


class ChatRequest(BaseModel):
    """Incoming chat message from the AI SDK frontend."""

    message: str
    session_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def sanitise_message(cls, v):
        v = sanitise_text(v)
        if len(v) > 5000:
            raise ValueError("Message too long (max 5000 chars)")
        if not v:
            raise ValueError("Message cannot be empty")
        return v
