"""
Robust LLM output parser — handles common format mismatches between
LLM JSON output and Pydantic model expectations.

Strategy:
1. Try direct Pydantic validation
2. On failure, apply field name mapping + type coercion
3. Fill missing required fields with sensible defaults
4. Retry validation
"""
import json
import logging
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


# Common field name variants LLMs use
FIELD_ALIASES: dict[str, list[str]] = {
    # Legal
    "legal_exposure_score": ["overall_exposure_score", "exposure_score", "overall_score", "risk_score"],
    "worst_case": ["worst_case_scenario", "worstCase", "worst_case_outcome"],
    "regulatory_gaps": ["regulatory_compliance_gaps", "compliance_gaps", "regulation_gaps"],
    "contract_formation_risk": ["contract_risk", "contract_formation", "contractFormationRisk"],
    "tort_exposure": ["tort_risk", "tortExposure", "negligence_exposure"],
    "key_uncertainties": ["uncertainties", "legal_uncertainties"],
    # Technical
    "technical_risk_score": ["overall_risk_score", "risk_score", "overall_score"],
    "factor_scores": ["risk_factor_scores", "scores"],
    "amplification_effects": ["amplification", "compounding_effects", "compound_effects"],
    "key_vulnerabilities": ["vulnerabilities", "critical_vulnerabilities"],
    # Mitigation
    "overall_mitigation_score": ["mitigation_score", "overall_score"],
    "axis_scores": ["mitigation_axes", "axes"],
    "quick_wins": ["quick_win_mitigations", "easy_wins"],
    # Pricing
    "premium_band": ["premium", "premium_range"],
    "premium_reasoning": ["pricing_reasoning", "reasoning"],
    "top_exposures": ["exposures", "key_exposures"],
    "data_gaps": ["information_gaps", "missing_data"],
    # Intake
    "existing_guardrails": ["guardrails", "safety_measures"],
    # Shared
    "confidence": ["confidence_score", "confidence_level"],
    "doctrine_assessments": ["legal_analysis", "doctrines", "assessments", "applicable_doctrines"],
    # DoctrineAssessment
    "doctrine_name": ["doctrine"],
    "reasoning": ["rationale", "justification", "explanation", "analysis_reasoning"],
    "exposure_level": ["risk_level", "exposure"],
    "applies": ["applicable", "is_applicable"],
    # RegulatoryGap
    "risk_if_non_compliant": ["non_compliance_risk", "penalty", "consequence"],
    # FactorScore
    "factor_name": ["risk_factor"],
    "missing_info": ["missing_information"],
    # MitigationAxisScore
    "present_mitigations": ["existing_mitigations"],
    "missing_mitigations": ["recommended_mitigations"],
    # RiskScenario
    "scenario_type": ["scenario"],
    "expected_loss_range": ["expected_loss", "loss_range", "financial_impact"],
    "mitigation_options": ["mitigation_recommendations"],
}


def _apply_aliases(data: dict) -> dict:
    """Remap known LLM field name variants to expected names."""
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical not in data:
            for alias in aliases:
                if alias in data:
                    data[canonical] = data.pop(alias)
                    break
    # Recurse into lists of dicts (e.g., doctrine_assessments)
    for key, val in data.items():
        if isinstance(val, list):
            data[key] = [_apply_aliases(item) if isinstance(item, dict) else item for item in val]
        elif isinstance(val, dict):
            data[key] = _apply_aliases(val)
    return data


def _coerce_types(data: dict) -> dict:
    """Fix common type mismatches: strings that should be lists, etc."""
    # String -> list coercion for fields that should be lists
    for key in ("existing_guardrails", "data_access", "key_risks_identified",
                "key_uncertainties", "key_vulnerabilities", "amplification_effects",
                "quick_wins", "data_gaps", "applicable_doctrines", "mitigation_options",
                "present_mitigations", "missing_mitigations", "critical_gaps"):
        if key in data and isinstance(data[key], str):
            data[key] = [s.strip() for s in data[key].split(",") if s.strip()]

    # Dict -> string coercion for list[str] fields where LLM returns list[dict]
    for key in ("key_risks_identified", "key_uncertainties", "key_vulnerabilities",
                "amplification_effects", "quick_wins", "data_gaps",
                "present_mitigations", "missing_mitigations", "critical_gaps",
                "existing_guardrails", "applicable_doctrines", "mitigation_options"):
        if key in data and isinstance(data[key], list):
            coerced = []
            for item in data[key]:
                if isinstance(item, dict):
                    # Extract first string value or join all values
                    vals = [str(v) for v in item.values() if v]
                    coerced.append("; ".join(vals) if vals else str(item))
                else:
                    coerced.append(item)
            data[key] = coerced

    # Recurse
    for key, val in data.items():
        if isinstance(val, list):
            data[key] = [_coerce_types(item) if isinstance(item, dict) else item for item in val]
        elif isinstance(val, dict):
            data[key] = _coerce_types(val)
    return data


# Default values for required fields when LLM omits them entirely
FIELD_DEFAULTS: dict[str, object] = {
    "confidence": 0.5,
    "legal_exposure_score": 0.5,
    "technical_risk_score": 0.5,
    "overall_mitigation_score": 0.3,
    "overall_risk_score": 0.5,
    "applies": True,
    "worst_case": "Not assessed — insufficient information",
    "reasoning": "See analysis details above",
    "regulatory_gaps": [],
    "contract_formation_risk": "Not assessed",
    "tort_exposure": "Not assessed",
    "key_uncertainties": ["Incomplete information provided"],
    "key_vulnerabilities": [],
    "amplification_effects": [],
    "quick_wins": [],
    "data_gaps": ["Assessment based on limited information"],
    "premium_reasoning": "Based on composite risk analysis",
    "executive_summary": "Risk assessment completed with available information.",
}


def _fill_defaults(data: dict, model: Type[BaseModel]) -> dict:
    """Fill missing required fields with defaults."""
    for field_name, field_info in model.model_fields.items():
        if field_name not in data and field_name in FIELD_DEFAULTS:
            data[field_name] = FIELD_DEFAULTS[field_name]

    # Recurse into nested models
    for field_name, field_info in model.model_fields.items():
        if field_name in data and isinstance(data[field_name], list):
            annotation = field_info.annotation
            # Check if it's a list of BaseModel subclasses
            if hasattr(annotation, "__args__"):
                inner_type = annotation.__args__[0] if annotation.__args__ else None
                if inner_type and isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                    data[field_name] = [
                        _fill_defaults(item, inner_type) if isinstance(item, dict) else item
                        for item in data[field_name]
                    ]
    return data


def parse_llm_output(raw_content: str, model: Type[T], agent_name: str) -> T:
    """
    Parse LLM output into a Pydantic model with robust error recovery.

    1. Extract JSON from markdown fences
    2. Try direct validation
    3. On failure: apply aliases, coerce types, fill defaults, retry
    """
    raw = extract_json(raw_content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{agent_name} returned invalid JSON: {e}") from e

    # Unwrap if LLM nested everything under a single wrapper key
    if isinstance(parsed, dict) and len(parsed) == 1:
        key = next(iter(parsed))
        val = parsed[key]
        if isinstance(val, dict) and key not in model.model_fields:
            parsed = val

    # First try: direct validation
    try:
        return model(**parsed)
    except ValidationError:
        pass  # Fall through to coercion

    # Second try: apply fixes
    parsed = _apply_aliases(parsed)
    parsed = _coerce_types(parsed)
    parsed = _fill_defaults(parsed, model)

    try:
        return model(**parsed)
    except ValidationError as e:
        logger.error("%s output failed validation after coercion: %s", agent_name, e)
        raise ValueError(f"{agent_name} output failed validation: {e}") from e
