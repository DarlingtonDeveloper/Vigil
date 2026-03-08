import { streamText, tool, convertToModelMessages, stepCountIs } from "ai";
import { anthropic } from "@ai-sdk/anthropic";
import { z } from "zod";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8080";
const API_KEY = process.env.BACKEND_API_KEY || "pg-demo-key-2025";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: anthropic("claude-sonnet-4-20250514"),
    system: `You are Vigil, an AI risk pricing engine for agentic AI deployments.

You help companies understand the legal and operational risk of deploying AI agents.

IMPORTANT WORKFLOW RULES:
- When a user describes their agentic system, ask AT MOST ONE round of clarifying questions covering any gaps, then IMMEDIATELY call the assessDeployment tool on the next turn. Do NOT ask multiple rounds of follow-up questions. Gather what you can from their description and fill in reasonable defaults for anything missing.
- If the user answers your clarifying questions, call assessDeployment IMMEDIATELY on that same turn. Do not ask for more details. Do not ask about contract terms, indemnification, or other tangential topics. Run the assessment with what you have.
- NEVER provide a "manual risk analysis" — always use the assessDeployment tool. If the tool fails, tell the user about the error.

Clarifying questions (ask all at once in a single message if needed):
- What tools/actions does the agent have access to?
- What data can it access? (PII, financial, health)
- What human oversight exists?
- Which jurisdictions does it operate in?

After assessment, explain results clearly:
- Lead with the risk score and premium band
- Highlight the top 3 exposures
- Explain which legal doctrines create the most risk
- Recommend the highest-impact mitigations
- Suggest specific "what if" scenarios to explore

When the user asks to run a specific risk scenario (data breach, hallucination, contract formation, etc.), use the runScenario tool with the session_id from the most recent assessment.

When discussing legal risk, be precise — cite specific doctrines (apparent authority, negligent misrepresentation, EU AI Act Article 6) rather than being generic.`,
    messages: await convertToModelMessages(messages),
    tools: {
      assessDeployment: tool({
        description:
          "Assess an agentic AI deployment for legal and operational risk. Use when the user describes their AI agent system.",
        inputSchema: z.object({
          description: z
            .string()
            .describe("Full description of the agentic deployment"),
          jurisdictions: z
            .array(z.string())
            .default(["UK"])
            .describe("Operating jurisdictions"),
          sector: z.string().optional().describe("Industry sector"),
        }),
        execute: async ({ description, jurisdictions, sector }) => {
          const res = await fetch(`${BACKEND_URL}/api/assess`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-API-Key": API_KEY,
            },
            body: JSON.stringify({ description, jurisdictions, sector }),
          });
          if (!res.ok) {
            const err = await res.text();
            return { error: `Assessment failed: ${err}` };
          }
          return await res.json();
        },
      }),

      runScenario: tool({
        description:
          "Run a risk scenario simulation against an existing assessment. Use after an assessment has been completed to explore specific failure modes.",
        inputSchema: z.object({
          session_id: z
            .string()
            .describe("The session_id from a completed assessment"),
          scenario_type: z
            .enum([
              "hallucination",
              "contract_formation",
              "data_breach",
              "scope_creep",
              "adversarial",
              "regulatory_breach",
            ])
            .describe("Type of risk scenario to simulate"),
        }),
        execute: async ({ session_id, scenario_type }) => {
          const res = await fetch(`${BACKEND_URL}/api/scenario`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-API-Key": API_KEY,
            },
            body: JSON.stringify({ session_id, scenario_type }),
          });
          if (!res.ok) {
            const err = await res.text();
            return { error: `Scenario failed: ${err}` };
          }
          return await res.json();
        },
      }),

      queryKnowledge: tool({
        description:
          "Query the legal knowledge graph for doctrines, regulations, risk factors, or mitigations.",
        inputSchema: z.object({
          query_type: z
            .enum([
              "doctrines",
              "regulations",
              "risk-factors",
              "stats",
              "full",
            ])
            .describe("Type of knowledge query"),
          jurisdiction: z
            .string()
            .optional()
            .describe("Jurisdiction filter"),
        }),
        execute: async ({ query_type, jurisdiction }) => {
          let url = `${BACKEND_URL}/api/knowledge/${query_type}`;
          if (jurisdiction) url += `?jurisdiction=${jurisdiction}`;
          const res = await fetch(url, {
            headers: { "X-API-Key": API_KEY },
          });
          if (!res.ok) return { error: "Knowledge query failed" };
          return await res.json();
        },
      }),

      getMitigations: tool({
        description:
          "Get recommended mitigations for a specific risk factor.",
        inputSchema: z.object({
          risk_factor: z
            .string()
            .describe(
              "Risk factor name, e.g. 'hallucination_risk', 'autonomy_level'"
            ),
        }),
        execute: async ({ risk_factor }) => {
          const res = await fetch(
            `${BACKEND_URL}/api/knowledge/mitigations/${encodeURIComponent(risk_factor)}`,
            { headers: { "X-API-Key": API_KEY } }
          );
          if (!res.ok) return { error: "Mitigation query failed" };
          return await res.json();
        },
      }),
    },
    stopWhen: stepCountIs(5),
  });

  return result.toUIMessageStreamResponse();
}
