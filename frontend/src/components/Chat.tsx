"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useRef, useState, useMemo } from "react";

/* no props needed — single-panel layout */

/* ── Constants ──────────────────────────── */

const TOOL_LABELS: Record<string, string> = {
  assessDeployment: "Running risk assessment",
  runScenario: "Simulating scenario",
  queryKnowledge: "Querying knowledge graph",
  getMitigations: "Finding mitigations",
};

const DEMO_SCENARIOS = [
  {
    title: "Customer Support Agent",
    desc: "AI agent with CRM access, customer email sending, and refund processing up to \u00A3500. UK jurisdiction with human-on-the-loop oversight.",
    tags: ["CRM", "Email", "Payments", "PII"],
    full: "We\u2019re deploying a customer support agent that can access our CRM, send emails to customers, and issue refunds up to \u00A3500. It uses Claude and operates in the UK. A general operator reviews flagged interactions but the agent handles most queries autonomously.",
  },
  {
    title: "Financial Advisor Bot",
    desc: "Investment recommendations, portfolio data access, and trade execution for retail customers. UK & EU regulated.",
    tags: ["Trading", "Portfolio", "Financial", "Regulated"],
    full: "We\u2019re building a financial advisor chatbot that provides investment recommendations to retail customers. It can access portfolio data and execute trades up to \u00A310,000. UK and EU regulated. Human approval required for trades over \u00A35,000.",
  },
  {
    title: "Internal Research Agent",
    desc: "Document summarisation and knowledge base Q&A. No external access, no PII, internal use only.",
    tags: ["Documents", "Internal", "Read-only"],
    full: "We have an internal research agent that summarises documents and answers questions from our knowledge base. No external access, no customer data, internal only. Used by our legal and compliance team.",
  },
];

const SCENARIO_ACTIONS = [
  {
    type: "data_breach",
    label: "Data Breach",
    prompt: "Run a data breach scenario on the current assessment. What would happen if the agent\u2019s data access was compromised?",
  },
  {
    type: "hallucination",
    label: "Hallucination",
    prompt: "Run a hallucination scenario. What\u2019s the risk if the agent generates false or misleading information?",
  },
  {
    type: "contract_formation",
    label: "Contract Formation",
    prompt: "Run a contract formation scenario. Could the agent inadvertently create binding commitments?",
  },
  {
    type: "scope_creep",
    label: "Scope Creep",
    prompt: "Run a scope creep scenario. What if the agent starts taking actions beyond its intended scope?",
  },
  {
    type: "adversarial",
    label: "Adversarial Attack",
    prompt: "Run an adversarial attack scenario. How vulnerable is this deployment to prompt injection or manipulation?",
  },
  {
    type: "regulatory_breach",
    label: "Regulatory Breach",
    prompt: "Run a regulatory breach scenario. Where are the biggest compliance gaps?",
  },
];

const KNOWLEDGE_ACTIONS = [
  {
    label: "Legal Doctrines",
    prompt: "What are the key legal doctrines that apply to this deployment? Show me the relevant doctrine assessments from the knowledge graph.",
  },
  {
    label: "Regulations",
    prompt: "Query the knowledge graph for all applicable regulations for this deployment\u2019s jurisdictions.",
  },
  {
    label: "Risk Factors",
    prompt: "Show me the full risk factor taxonomy from the knowledge graph.",
  },
];

/* Scenario-specific quick replies: keyed by scenario index, each has rounds of answers */
const SCENARIO_REPLIES: Record<number, string[][]> = {
  // Customer Support Agent
  0: [
    [
      "It reads and writes to our CRM, sends customer emails, and processes refunds up to \u00A3500 via Stripe. Customer PII is accessible. A human operator reviews flagged interactions only. UK jurisdiction, e-commerce sector.",
      "CRM read/write, email sending, and Stripe refunds up to \u00A3500. It accesses customer names, email addresses, and order history. Operator reviews 10% of interactions randomly. UK only, retail.",
    ],
    [
      "Yes, that covers it. Please run the assessment now.",
      "We also have content filtering on outbound emails and a kill switch. Please run the assessment.",
    ],
  ],
  // Financial Advisor Bot
  1: [
    [
      "It accesses portfolio data, trade history, and account balances. It can execute trades up to \u00A310,000 autonomously. Human approval required above \u00A35,000. UK and EU, FCA-authorised. Financial services sector.",
      "Read access to portfolios, client risk profiles, and market data. Trade execution up to \u00A35,000. All trades logged and audited daily. UK and EU regulated, FCA permissions held.",
    ],
    [
      "Yes, that\u2019s everything. Please run the assessment.",
      "We also have real-time position limits and compliance checks before each trade. Run the assessment please.",
    ],
  ],
  // Internal Research Agent
  2: [
    [
      "Read-only access to our internal knowledge base and document store. No customer data, no PII, no external access. Fully autonomous with comprehensive logging. UK only, used by legal and compliance team.",
      "It searches and summarises internal policy documents and legal memos. No access to customer data or external systems. Used internally by 15 people. UK jurisdiction.",
    ],
    [
      "That covers everything. Please run the assessment.",
      "We also have query logging and monthly access reviews. Go ahead with the assessment.",
    ],
  ],
};

/* Generic quick replies for free-text conversations (not demo scenarios) */
const GENERIC_REPLIES: string[][] = [
  [
    "It reads and writes to our CRM, sends customer emails, and processes refunds up to \u00A3500. Customer PII accessible. Human reviews flagged interactions. UK, e-commerce.",
    "Read-only access to internal databases. No external actions, no PII. Internal use only. UK jurisdiction.",
    "Full CRUD on customer records, API calls to third-party services, outbound emails. Financial data access with human approval over \u00A35,000. UK and EU, financial services.",
  ],
  [
    "Yes, that covers everything. Please run the assessment.",
    "We also have content filtering, rate limiting, and a kill switch for the operator. Run the assessment.",
  ],
];

/* ── Component ──────────────────────────── */

export default function Chat() {
  const [input, setInput] = useState("");
  const [hasAssessment, setHasAssessment] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const transport = useMemo(
    () => new DefaultChatTransport({ api: "/api/chat" }),
    []
  );

  const { messages, sendMessage, status } = useChat({
    transport,
    onFinish: ({ message }) => {
      try {
        for (const part of message.parts) {
          if (
            part.type === "tool-invocation" &&
            "toolName" in part &&
            part.toolName === "assessDeployment"
          ) {
            const output =
              "output" in part
                ? part.output
                : "result" in part
                  ? (part as Record<string, unknown>).result
                  : null;
            if (
              output &&
              typeof output === "object" &&
              !("error" in (output as Record<string, unknown>))
            ) {
              setHasAssessment(true);
            }
          }
        }
      } catch (e) {
        console.error("Assessment extraction error:", e);
      }
    },
  });

  const isActive = status === "submitted" || status === "streaming";

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, hasAssessment]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isActive) return;
    const text = input;
    setInput("");
    sendMessage({ text });
  };

  const send = (text: string) => {
    if (isActive) return;
    sendMessage({ text });
  };

  return (
    <>
      {/* ── Messages / Demo Cards ──────── */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        {messages.length === 0 ? (
          /* ── Initial: Demo scenario cards ── */
          <div
            className="space-y-6"
            style={{
              animation: "fade-up 0.25s cubic-bezier(0.25, 1, 0.5, 1)",
            }}
          >
            <div>
              <p className="text-[11px] uppercase tracking-[0.2em] text-muted font-medium mb-1">
                Demo Scenarios
              </p>
              <p className="text-[13px] text-muted/60">
                Click to run a full risk assessment pipeline
              </p>
            </div>

            <div className="space-y-3">
              {DEMO_SCENARIOS.map((s, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setSelectedScenario(i);
                    send(s.full);
                  }}
                  disabled={isActive}
                  className="group w-full text-left rounded-lg bg-surface/50 border border-edge/50 hover:border-gold/25 transition-all duration-150 disabled:opacity-40 overflow-hidden"
                  style={{
                    animation: `fade-up 0.25s cubic-bezier(0.25, 1, 0.5, 1) ${0.05 + i * 0.06}s backwards`,
                  }}
                >
                  <div className="px-4 py-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <span className="text-[15px] font-medium text-primary/90 group-hover:text-gold transition-colors duration-150">
                          {s.title}
                        </span>
                        <p className="text-[13px] text-secondary/70 mt-1.5 leading-relaxed">
                          {s.desc}
                        </p>
                      </div>
                      <span className="flex-shrink-0 mt-0.5 text-muted/40 group-hover:text-gold/60 transition-colors duration-150">
                        <svg
                          className="w-4 h-4"
                          viewBox="0 0 16 16"
                          fill="none"
                        >
                          <path
                            d="M3 8H13M13 8L8.5 3.5M13 8L8.5 12.5"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </span>
                    </div>
                    <div className="flex gap-1.5 mt-3">
                      {s.tags.map((tag) => (
                        <span
                          key={tag}
                          className="text-[10px] uppercase tracking-wider text-muted/60 bg-elevated/60 border border-edge/30 rounded px-2 py-0.5"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              ))}
            </div>

            <div className="pt-2">
              <p className="text-[11px] uppercase tracking-[0.2em] text-muted/40 font-medium">
                Or type your own below
              </p>
            </div>
          </div>
        ) : (
          /* ── Active: Messages + follow-up actions ── */
          <div className="space-y-4">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                style={{
                  animation:
                    "fade-up 0.2s cubic-bezier(0.25, 1, 0.5, 1) backwards",
                }}
              >
                <div
                  className={`max-w-[88%] rounded-lg px-4 py-3 ${
                    m.role === "user"
                      ? "bg-gold/8 border border-gold/12"
                      : "bg-surface/70 border border-edge/50"
                  }`}
                >
                  {m.parts.map((part, i) => {
                    /* Text */
                    if (part.type === "text" && part.text) {
                      return (
                        <div
                          key={i}
                          className="text-[15px] leading-[1.7] whitespace-pre-wrap"
                        >
                          {part.text}
                          {isActive &&
                            m.role === "assistant" &&
                            i === m.parts.length - 1 && (
                              <span
                                className="inline-block w-[2px] h-[14px] bg-gold/60 ml-0.5 align-middle"
                                style={{
                                  animation: "blink 1s step-end infinite",
                                }}
                              />
                            )}
                        </div>
                      );
                    }

                    /* Tool invocation */
                    if (part.type === "tool-invocation" && "toolName" in part) {
                      const label =
                        TOOL_LABELS[String(part.toolName)] ||
                        String(part.toolName);
                      const state =
                        "state" in part ? String(part.state) : "";
                      const done =
                        state === "result" || state === "output-available";
                      const error = state === "output-error";

                      return (
                        <div
                          key={i}
                          className={`my-2 flex items-center gap-2.5 text-[13px] rounded-md py-2.5 px-3.5 border ${
                            done
                              ? "bg-safe/4 border-safe/10 text-safe/80"
                              : error
                                ? "bg-danger/4 border-danger/10 text-danger/80"
                                : "bg-gold/4 border-gold/10 text-gold/70"
                          }`}
                          style={{ fontFamily: MONO }}
                        >
                          {done ? (
                            <IconCheck />
                          ) : error ? (
                            <IconX />
                          ) : (
                            <IconSpinner />
                          )}
                          <span>
                            {done
                              ? `${label} \u2014 complete`
                              : error
                                ? `${label} \u2014 failed`
                                : `${label}\u2026`}
                          </span>
                        </div>
                      );
                    }

                    return null;
                  })}
                </div>
              </div>
            ))}

            {/* ── Quick reply buttons ──────── */}
            {!isActive &&
              !hasAssessment &&
              messages.length > 0 &&
              messages[messages.length - 1].role === "assistant" &&
              (() => {
                const assistantTurns = messages.filter(
                  (m) =>
                    m.role === "assistant" &&
                    m.parts.some((p) => p.type === "text" && p.text)
                ).length;
                // Use scenario-specific replies if a demo was selected, else generic
                const replyBank =
                  selectedScenario !== null
                    ? SCENARIO_REPLIES[selectedScenario]
                    : GENERIC_REPLIES;
                if (!replyBank) return null;
                const roundIdx = Math.min(
                  assistantTurns - 1,
                  replyBank.length - 1
                );
                const replySet = replyBank[roundIdx];
                if (!replySet) return null;
                return (
                  <div
                    className="flex flex-wrap gap-1.5 mt-1"
                    style={{
                      animation:
                        "fade-up 0.2s cubic-bezier(0.25, 1, 0.5, 1) 0.1s backwards",
                    }}
                  >
                    {replySet.map((reply, ri) => (
                      <button
                        key={ri}
                        onClick={() => send(reply)}
                        className="text-left text-[13px] leading-relaxed text-secondary/80 hover:text-gold bg-surface/40 hover:bg-gold/4 border border-edge/40 hover:border-gold/20 rounded-md px-3.5 py-2.5 transition-colors duration-150 max-w-full"
                      >
                        {reply}
                      </button>
                    ))}
                  </div>
                );
              })()}

            {/* Loading dots */}
            {isActive &&
              messages.length > 0 &&
              messages[messages.length - 1].role === "user" && (
                <div
                  className="flex justify-start"
                  style={{
                    animation:
                      "fade-up 0.2s cubic-bezier(0.25, 1, 0.5, 1) backwards",
                  }}
                >
                  <div className="bg-surface/70 border border-edge/50 rounded-lg px-4 py-3">
                    <div className="flex gap-1.5">
                      {[0, 1, 2].map((d) => (
                        <span
                          key={d}
                          className="w-1.5 h-1.5 rounded-full bg-gold/40"
                          style={{
                            animation: `pulse-dot 1.4s ease-in-out ${d * 0.2}s infinite`,
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              )}

            {/* ── Follow-up workflow buttons ── */}
            {hasAssessment && !isActive && (
              <div
                className="pt-4 pb-2 space-y-4"
                style={{
                  animation:
                    "fade-up 0.25s cubic-bezier(0.25, 1, 0.5, 1) backwards",
                }}
              >
                {/* Scenario buttons */}
                <div>
                  <p className="text-[11px] uppercase tracking-[0.2em] text-muted font-medium mb-2.5">
                    Run Scenario
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {SCENARIO_ACTIONS.map((s) => (
                      <button
                        key={s.type}
                        onClick={() => send(s.prompt)}
                        className="group text-left px-3.5 py-3 rounded-md bg-surface/40 border border-edge/40 hover:border-gold/20 transition-colors duration-150"
                      >
                        <span className="block text-[13px] text-secondary/80 group-hover:text-gold transition-colors duration-150 leading-tight">
                          {s.label}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Knowledge buttons */}
                <div>
                  <p className="text-[11px] uppercase tracking-[0.2em] text-muted font-medium mb-2.5">
                    Knowledge Graph
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {KNOWLEDGE_ACTIONS.map((k) => (
                      <button
                        key={k.label}
                        onClick={() => send(k.prompt)}
                        className="group text-left px-3.5 py-3 rounded-md bg-surface/40 border border-edge/40 hover:border-gold/20 transition-colors duration-150"
                      >
                        <span className="block text-[13px] text-secondary/80 group-hover:text-gold transition-colors duration-150 leading-tight">
                          {k.label}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* ── Input ───────────────────────── */}
      <div className="px-8 py-5 border-t border-edge/60">
        <form onSubmit={handleSubmit} className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your AI agent deployment\u2026"
            disabled={isActive}
            className="w-full bg-surface/70 border border-edge/70 rounded-lg px-4 py-3.5 pr-12 text-[15px] text-primary placeholder:text-muted/50 focus:outline-none focus:border-gold/25 focus:ring-1 focus:ring-gold/8 transition-all duration-150 disabled:opacity-40"
          />
          <button
            type="submit"
            disabled={isActive || !input.trim()}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded text-muted hover:text-gold disabled:opacity-20 disabled:hover:text-muted transition-colors duration-150"
          >
            <svg className="w-5 h-5" viewBox="0 0 16 16" fill="none">
              <path
                d="M3 8H13M13 8L8.5 3.5M13 8L8.5 12.5"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </form>
      </div>
    </>
  );
}

/* ── Constants & Icons ──────────────────── */

const MONO = "var(--font-mono, 'JetBrains Mono'), monospace";

function IconCheck() {
  return (
    <svg className="w-3 h-3 flex-shrink-0" viewBox="0 0 16 16" fill="none">
      <path
        d="M3.5 8.5L6.5 11.5L12.5 4.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconX() {
  return (
    <svg className="w-3 h-3 flex-shrink-0" viewBox="0 0 16 16" fill="none">
      <path
        d="M4.5 4.5L11.5 11.5M11.5 4.5L4.5 11.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconSpinner() {
  return (
    <svg
      className="w-3 h-3 flex-shrink-0"
      viewBox="0 0 16 16"
      fill="none"
      style={{ animation: "spin-slow 1s linear infinite" }}
    >
      <circle
        cx="8"
        cy="8"
        r="5.5"
        stroke="currentColor"
        strokeWidth="1.5"
        opacity="0.2"
      />
      <path
        d="M8 2.5A5.5 5.5 0 0 1 13.5 8"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
