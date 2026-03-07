import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import FaultLineState
from app.agents.intake import run_intake
from app.agents.legal import run_legal_analysis
from app.agents.technical import run_technical_analysis
from app.agents.mitigation import run_mitigation_analysis
from app.agents.pricing import run_pricing
from app.db.client import db
from app.db.queries import log_audit
from app.tracing.opik_setup import get_opik_tracer
from app.tracing.langsmith_setup import get_langsmith_run_config


async def intake_node(state: FaultLineState) -> dict:
    """Parse deployment description into structured profile."""
    sid = state["session_id"]
    await log_audit(sid, "intake", "start")

    profile = await run_intake(
        state["description"], state["jurisdictions"],
        state.get("sector"), sid
    )
    profile_dict = profile.model_dump()

    # Store profile
    await db.query("""
        CREATE deployment_profile SET
            session_id = $sid,
            agent_description = $desc,
            tools = $tools,
            data_access = $data,
            autonomy_level = $autonomy,
            output_reach = $reach,
            sector = $sector,
            jurisdictions = $jurisdictions,
            human_oversight_model = $oversight,
            existing_guardrails = $guardrails
    """, {
        "sid": sid, "desc": profile_dict.get("agent_description", ""),
        "tools": profile_dict.get("tools", []),
        "data": profile_dict.get("data_access", []),
        "autonomy": profile_dict.get("autonomy_level", ""),
        "reach": profile_dict.get("output_reach", ""),
        "sector": profile_dict.get("sector", ""),
        "jurisdictions": profile_dict.get("jurisdictions", []),
        "oversight": profile_dict.get("human_oversight_model"),
        "guardrails": profile_dict.get("existing_guardrails", []),
    })

    await log_audit(sid, "intake", "complete",
                    output_data={"tools": len(profile.tools), "autonomy": profile.autonomy_level})
    return {"deployment_profile": profile_dict, "current_step": "intake_complete"}


async def fetch_knowledge_node(state: FaultLineState) -> dict:
    """Fetch relevant knowledge graph data from SurrealDB for the agents."""
    sid = state["session_id"]
    await log_audit(sid, "knowledge_fetch", "start")
    profile = state["deployment_profile"]
    jurisdictions = profile.get("jurisdictions", ["UK"])

    # Determine relevant legal domains based on profile
    domains = ["contract_law", "tort_law", "data_protection", "regulatory_compliance"]
    if profile.get("sector") in ["financial", "healthcare", "legal"]:
        domains.append("employment_law")
    if any(t.get("action_type") == "communicate" for t in profile.get("tools", [])):
        domains.append("intellectual_property")

    doctrines = await db.get_applicable_doctrines(jurisdictions, domains)
    regulations = await db.get_applicable_regulations(jurisdictions)
    risk_factors = await db.query("SELECT * FROM risk_factor ORDER BY weight DESC")
    mitigations = await db.query("SELECT * FROM mitigation")
    mit_edges = await db.query("""
        SELECT in.name AS mitigation, out.name AS risk_factor, reduction, conditions
        FROM mitigates
    """)

    # Flatten results
    def flatten(r):
        if r and isinstance(r, list) and r[0] and isinstance(r[0], dict):
            return r[0].get("result", r) if "result" in r[0] else r
        return r or []

    await log_audit(sid, "knowledge_fetch", "complete")

    return {
        "applicable_doctrines": flatten(doctrines),
        "applicable_regulations": flatten(regulations),
        "risk_factors": flatten(risk_factors),
        "available_mitigations": flatten(mitigations),
        "mitigation_edges": flatten(mit_edges),
        "current_step": "knowledge_fetched",
    }


async def legal_node(state: FaultLineState) -> dict:
    sid = state["session_id"]
    await log_audit(sid, "legal", "start")

    analysis = await run_legal_analysis(
        state["deployment_profile"], state["applicable_doctrines"],
        state["applicable_regulations"], sid
    )

    await log_audit(sid, "legal", "complete",
                    output_data={"exposure_score": analysis.legal_exposure_score})
    return {"legal_analysis": analysis.model_dump(), "current_step": "legal_complete"}


async def technical_node(state: FaultLineState) -> dict:
    sid = state["session_id"]
    await log_audit(sid, "technical", "start")

    analysis = await run_technical_analysis(
        state["deployment_profile"], state["risk_factors"], sid
    )

    await log_audit(sid, "technical", "complete",
                    output_data={"risk_score": analysis.technical_risk_score})
    return {"technical_analysis": analysis.model_dump(), "current_step": "technical_complete"}


async def mitigation_node(state: FaultLineState) -> dict:
    sid = state["session_id"]
    await log_audit(sid, "mitigation", "start")

    analysis = await run_mitigation_analysis(
        state["deployment_profile"], state["available_mitigations"],
        state["mitigation_edges"], sid
    )

    await log_audit(sid, "mitigation", "complete",
                    output_data={"mitigation_score": analysis.overall_mitigation_score})
    return {"mitigation_analysis": analysis.model_dump(), "current_step": "mitigation_complete"}


async def pricing_node(state: FaultLineState) -> dict:
    sid = state["session_id"]
    await log_audit(sid, "pricing", "start")

    price = await run_pricing(
        state["legal_analysis"], state["technical_analysis"],
        state["mitigation_analysis"], state["deployment_profile"], sid
    )

    # Store risk score
    await db.query("""
        CREATE risk_score SET
            session_id = $sid,
            technical_risk = $tech,
            legal_exposure = $legal,
            market_conditions = 0.5,
            mitigation_profile = $mit,
            overall_risk = $overall,
            premium_estimate = $premium,
            confidence = $conf,
            reasoning = $reasoning,
            top_exposures = $exposures,
            recommendations = $recs,
            created_at = time::now()
    """, {
        "sid": sid,
        "tech": price.technical_risk,
        "legal": price.legal_exposure,
        "mit": state["mitigation_analysis"],
        "overall": price.overall_risk_score,
        "premium": price.premium_band,
        "conf": price.confidence,
        "reasoning": price.executive_summary,
        "exposures": price.top_exposures,
        "recs": price.recommendations,
    })

    # Store scenarios
    for scenario in price.scenarios:
        await db.query("""
            CREATE risk_scenario SET
                session_id = $sid,
                scenario_type = $type,
                description = $desc,
                probability = $prob,
                severity = $sev,
                expected_loss = $loss,
                applicable_doctrines = $doctrines,
                mitigation_options = $mits,
                created_at = time::now()
        """, {
            "sid": sid,
            "type": scenario.scenario_type,
            "desc": f"{scenario.probability} probability, {scenario.severity} severity",
            "prob": 0.5, "sev": 0.5,  # numeric proxies
            "loss": scenario.expected_loss_range,
            "doctrines": scenario.applicable_doctrines,
            "mits": scenario.mitigation_options,
        })

    # Update session
    await db.query(
        "UPDATE assessment SET status = 'completed', updated_at = time::now() WHERE session_id = $sid",
        {"sid": sid}
    )

    await log_audit(sid, "pricing", "complete",
                    output_data={"risk_score": price.overall_risk_score, "premium": price.premium_band})

    return {"risk_price": price.model_dump(), "current_step": "complete"}


def build_workflow():
    workflow = StateGraph(FaultLineState)

    workflow.add_node("intake", intake_node)
    workflow.add_node("fetch_knowledge", fetch_knowledge_node)
    workflow.add_node("legal", legal_node)
    workflow.add_node("technical", technical_node)
    workflow.add_node("mitigation", mitigation_node)
    workflow.add_node("pricing", pricing_node)

    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "fetch_knowledge")
    workflow.add_edge("fetch_knowledge", "legal")
    workflow.add_edge("legal", "technical")
    workflow.add_edge("technical", "mitigation")
    workflow.add_edge("mitigation", "pricing")
    workflow.add_edge("pricing", END)

    return workflow.compile(checkpointer=MemorySaver())


async def run_assessment(description: str, jurisdictions: list[str], sector: str | None,
                         user_id: str) -> dict:
    session_id = str(uuid.uuid4())

    if not db._conn:
        await db.connect()

    await db.query("""
        CREATE assessment SET
            session_id = $sid, user_id = $uid, deployment_description = $desc,
            status = 'running', created_at = time::now(), updated_at = time::now()
    """, {"sid": session_id, "uid": user_id, "desc": description})

    initial_state: FaultLineState = {
        "session_id": session_id, "user_id": user_id,
        "description": description, "jurisdictions": jurisdictions, "sector": sector,
        "deployment_profile": None, "applicable_doctrines": [], "applicable_regulations": [],
        "risk_factors": [], "available_mitigations": [], "mitigation_edges": [],
        "legal_analysis": None, "technical_analysis": None,
        "mitigation_analysis": None, "risk_price": None,
        "legal_quality_score": 0.0, "technical_quality_score": 0.0,
        "pricing_quality_score": 0.0, "current_step": "started", "error": None,
    }

    graph = build_workflow()
    opik_tracer = get_opik_tracer(session_id, ["full-assessment"])
    config = {**get_langsmith_run_config(session_id, "full-assessment"),
              "configurable": {"thread_id": session_id}, "callbacks": [opik_tracer]}

    try:
        return await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        await db.query("UPDATE assessment SET status = 'failed', updated_at = time::now() WHERE session_id = $sid",
                       {"sid": session_id})
        raise
