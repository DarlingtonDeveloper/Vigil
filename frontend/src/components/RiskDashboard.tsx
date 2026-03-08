"use client";

import { useEffect, useState } from "react";

/* ── Types ──────────────────────────────── */

interface Exposure {
  exposure?: string;
  name?: string;
  severity?: string;
  mitigation_available?: string;
}

interface Scenario {
  scenario_type: string;
  probability: string;
  severity: string;
  expected_loss_range: string;
  applicable_doctrines?: string[];
}

interface Recommendation {
  priority: string;
  action: string;
  reasoning?: string;
  impact?: string;
}

interface RiskPrice {
  overall_risk_score: number;
  premium_band: string;
  confidence: number;
  technical_risk: number;
  legal_exposure: number;
  mitigation_effectiveness: number;
  executive_summary: string;
  top_exposures?: Exposure[];
  scenarios?: Scenario[];
  recommendations?: Recommendation[];
  data_gaps?: string[];
}

const MONO = "var(--font-mono, 'JetBrains Mono'), monospace";
const ARC_LEN = 251.33; // pi * 80

/* ── Helpers ────────────────────────────── */

function riskText(s: number) {
  if (s < 0.35) return "text-safe";
  if (s < 0.65) return "text-warn";
  return "text-danger";
}

function riskBorder(s: number) {
  if (s < 0.35) return "border-safe/15";
  if (s < 0.65) return "border-warn/15";
  return "border-danger/15";
}

function riskBg(s: number) {
  if (s < 0.35) return "bg-safe/[0.03]";
  if (s < 0.65) return "bg-warn/[0.03]";
  return "bg-danger/[0.03]";
}

function priorityStyle(p: string) {
  if (p === "critical") return "text-danger bg-danger/8 border-danger/12";
  if (p === "high") return "text-warn bg-warn/8 border-warn/12";
  if (p === "medium") return "text-gold bg-gold/8 border-gold/12";
  return "text-secondary bg-surface border-edge";
}

function severityStyle(s: string) {
  if (s === "critical" || s === "high")
    return "text-danger bg-danger/6 border-danger/10";
  if (s === "medium") return "text-warn bg-warn/6 border-warn/10";
  return "text-muted bg-surface border-edge";
}

/* ── Section label ──────────────────────── */

function Label({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] uppercase tracking-[0.2em] text-muted font-medium mb-4">
      {children}
    </p>
  );
}

/* ── Card wrapper ───────────────────────── */

function Card({
  children,
  delay = 0,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <div
      className={`bg-surface/50 rounded-lg border border-edge/40 p-5 ${className}`}
      style={{
        animation: `fade-up 0.3s cubic-bezier(0.25, 1, 0.5, 1) ${delay}s backwards`,
      }}
    >
      {children}
    </div>
  );
}

/* ── Main component ─────────────────────── */

export default function RiskDashboard({
  data,
}: {
  data: Record<string, unknown>;
}) {
  const price = data?.risk_price as RiskPrice | undefined;
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(t);
  }, []);

  if (!price) return null;

  const score = price.overall_risk_score ?? 0;
  const pct = Math.round(score * 100);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="px-6 py-6 space-y-4 max-w-[640px]">
        {/* ── Gauge hero ──────────────────── */}
        <div
          className={`rounded-lg border ${riskBorder(score)} ${riskBg(score)} p-6`}
          style={{
            animation:
              "fade-up 0.3s cubic-bezier(0.25, 1, 0.5, 1) backwards",
          }}
        >
          <div className="flex items-start justify-between gap-6">
            {/* SVG arc gauge */}
            <div className="flex flex-col items-center flex-shrink-0">
              <svg viewBox="0 0 200 115" className="w-[180px]">
                <defs>
                  <linearGradient
                    id="gauge"
                    x1="0%"
                    y1="0%"
                    x2="100%"
                    y2="0%"
                  >
                    <stop offset="0%" stopColor="#34d399" />
                    <stop offset="45%" stopColor="#f59e0b" />
                    <stop offset="100%" stopColor="#ef4444" />
                  </linearGradient>
                </defs>
                {/* Track */}
                <path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke="#1f2133"
                  strokeWidth="5"
                  strokeLinecap="round"
                />
                {/* Fill */}
                <path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke="url(#gauge)"
                  strokeWidth="5"
                  strokeLinecap="round"
                  strokeDasharray={ARC_LEN}
                  strokeDashoffset={
                    mounted ? ARC_LEN * (1 - score) : ARC_LEN
                  }
                  style={{
                    transition: "stroke-dashoffset 1s cubic-bezier(0.25, 1, 0.5, 1) 0.2s",
                  }}
                />
                {/* Score number */}
                <text
                  x="100"
                  y="76"
                  textAnchor="middle"
                  fill="currentColor"
                  className={riskText(score)}
                  style={{
                    fontFamily: MONO,
                    fontSize: "36px",
                    fontWeight: 600,
                    animation: mounted
                      ? "score-reveal 0.4s cubic-bezier(0.25, 1, 0.5, 1) 0.3s backwards"
                      : "none",
                  }}
                >
                  {pct}
                </text>
                <text
                  x="100"
                  y="95"
                  textAnchor="middle"
                  fill="#464862"
                  style={{ fontSize: "9px", letterSpacing: "0.15em" }}
                >
                  RISK SCORE
                </text>
              </svg>
            </div>

            {/* Premium & confidence */}
            <div className="text-right space-y-4 pt-1">
              <div>
                <p className="text-[10px] uppercase tracking-[0.15em] text-muted mb-1">
                  Premium Band
                </p>
                <p
                  className="text-base font-semibold text-primary"
                  style={{ fontFamily: MONO }}
                >
                  {price.premium_band}
                </p>
              </div>
              <div>
                <p className="text-[10px] uppercase tracking-[0.15em] text-muted mb-1">
                  Confidence
                </p>
                <p
                  className="text-[13px] text-secondary tabular-nums"
                  style={{ fontFamily: MONO }}
                >
                  {((price.confidence ?? 0) * 100).toFixed(0)}%
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ── Score breakdown ─────────────── */}
        <div
          className="grid grid-cols-3 gap-3"
          style={{
            animation:
              "fade-up 0.3s cubic-bezier(0.25, 1, 0.5, 1) 0.08s backwards",
          }}
        >
          {[
            { label: "Technical Risk", value: price.technical_risk },
            { label: "Legal Exposure", value: price.legal_exposure },
            {
              label: "Mitigation",
              value: price.mitigation_effectiveness,
              invert: true,
            },
          ].map(({ label, value, invert }) => {
            const v = value ?? 0;
            return (
              <div
                key={label}
                className="bg-surface/50 rounded-lg border border-edge/40 p-4"
              >
                <p className="text-[10px] uppercase tracking-[0.15em] text-muted mb-2">
                  {label}
                </p>
                <p
                  className={`text-xl font-semibold tabular-nums ${riskText(invert ? 1 - v : v)}`}
                  style={{ fontFamily: MONO }}
                >
                  {(v * 100).toFixed(0)}
                  <span className="text-[11px] text-muted ml-0.5">%</span>
                </p>
              </div>
            );
          })}
        </div>

        {/* ── Executive summary ───────────── */}
        {price.executive_summary && (
          <Card delay={0.14}>
            <Label>Executive Summary</Label>
            <p className="text-[13px] text-secondary leading-[1.75]">
              {price.executive_summary}
            </p>
          </Card>
        )}

        {/* ── Top exposures ───────────────── */}
        {price.top_exposures && price.top_exposures.length > 0 && (
          <Card delay={0.2}>
            <Label>Top Exposures</Label>
            <div className="space-y-3">
              {price.top_exposures.map((exp, i) => (
                <div
                  key={i}
                  className="flex items-start justify-between gap-3"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] text-primary/85">
                      {exp.exposure || exp.name}
                    </p>
                    {exp.mitigation_available && (
                      <p className="text-[11px] text-muted mt-0.5">
                        Mitigation: {exp.mitigation_available}
                      </p>
                    )}
                  </div>
                  <span
                    className={`flex-shrink-0 text-[10px] font-medium px-2 py-0.5 rounded border ${severityStyle(exp.severity || "")}`}
                  >
                    {exp.severity}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* ── Risk scenarios ──────────────── */}
        {price.scenarios && price.scenarios.length > 0 && (
          <Card delay={0.26}>
            <Label>Risk Scenarios</Label>
            <div className="space-y-4">
              {price.scenarios.map((s, i) => (
                <div
                  key={i}
                  className="border-l-[2px] border-gold/20 pl-4 space-y-1"
                >
                  <p className="text-[13px] font-medium text-primary/85">
                    {s.scenario_type.replace(/_/g, " ")}
                  </p>
                  <div
                    className="flex gap-4 text-[11px] text-muted tabular-nums"
                    style={{ fontFamily: MONO }}
                  >
                    <span>P: {s.probability}</span>
                    <span>S: {s.severity}</span>
                  </div>
                  <p className="text-[11px] text-gold/50">
                    Expected loss: {s.expected_loss_range}
                  </p>
                  {s.applicable_doctrines &&
                    s.applicable_doctrines.length > 0 && (
                      <p className="text-[11px] text-muted/60">
                        {s.applicable_doctrines.join(" \u00B7 ")}
                      </p>
                    )}
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* ── Recommendations ─────────────── */}
        {price.recommendations && price.recommendations.length > 0 && (
          <Card delay={0.32}>
            <Label>Recommendations</Label>
            <div className="space-y-3">
              {price.recommendations.slice(0, 6).map((r, i) => (
                <div key={i} className="flex items-start gap-3">
                  <span
                    className={`flex-shrink-0 text-[10px] font-medium px-1.5 py-0.5 rounded border mt-0.5 ${priorityStyle(r.priority)}`}
                  >
                    {r.priority}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-[13px] text-primary/80">{r.action}</p>
                    {r.impact && (
                      <p className="text-[11px] text-gold/40 mt-0.5">
                        Impact: {r.impact}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* ── Data gaps ───────────────────── */}
        {price.data_gaps && price.data_gaps.length > 0 && (
          <Card delay={0.38} className="mb-4">
            <Label>Information Gaps</Label>
            <p className="text-[11px] text-muted/60 mb-3">
              Addressing these gaps would improve assessment confidence.
            </p>
            <div className="space-y-1.5">
              {price.data_gaps.map((gap, i) => (
                <div key={i} className="flex items-start gap-2">
                  <span className="text-gold/25 text-[11px] mt-px">&mdash;</span>
                  <p className="text-[12px] text-secondary/70">{gap}</p>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}
