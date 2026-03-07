"use client";

interface RiskDashboardProps {
  data: Record<string, unknown>;
}

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

export default function RiskDashboard({ data }: RiskDashboardProps) {
  const price = data?.risk_price as RiskPrice | undefined;
  if (!price) return null;

  const riskColor = (score: number) => {
    if (score < 0.3) return "text-green-400";
    if (score < 0.6) return "text-amber-400";
    return "text-red-400";
  };

  const riskBg = (score: number) => {
    if (score < 0.3) return "bg-green-400/10 border-green-400/20";
    if (score < 0.6) return "bg-amber-400/10 border-amber-400/20";
    return "bg-red-400/10 border-red-400/20";
  };

  const priorityColor = (priority: string) => {
    if (priority === "critical") return "text-red-400";
    if (priority === "high") return "text-amber-400";
    if (priority === "medium") return "text-yellow-400";
    return "text-gray-400";
  };

  return (
    <div className="p-4 space-y-6">
      {/* Overall Risk Score */}
      <div
        className={`rounded-lg border p-6 ${riskBg(price.overall_risk_score)}`}
      >
        <div className="flex justify-between items-start">
          <div>
            <p className="text-sm text-gray-400">Overall Risk Score</p>
            <p
              className={`text-5xl font-bold mt-1 ${riskColor(
                price.overall_risk_score
              )}`}
            >
              {(price.overall_risk_score * 100).toFixed(0)}
            </p>
            <p className="text-xs text-gray-500 mt-1">out of 100</p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-400">Estimated Premium</p>
            <p className="text-lg font-semibold text-gray-200 mt-1">
              {price.premium_band}
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Confidence: {(price.confidence * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>

      {/* Score Breakdown */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Technical Risk", value: price.technical_risk },
          { label: "Legal Exposure", value: price.legal_exposure },
          {
            label: "Mitigation Score",
            value: price.mitigation_effectiveness,
            invert: true,
          },
        ].map(({ label, value, invert }) => (
          <div
            key={label}
            className="bg-gray-900 rounded-lg p-4 border border-gray-800"
          >
            <p className="text-xs text-gray-500 mb-1">{label}</p>
            <p
              className={`text-2xl font-bold ${riskColor(
                invert ? 1 - (value || 0) : value || 0
              )}`}
            >
              {((value || 0) * 100).toFixed(0)}%
            </p>
          </div>
        ))}
      </div>

      {/* Executive Summary */}
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
        <p className="text-sm font-medium text-gray-300 mb-2">
          Executive Summary
        </p>
        <p className="text-sm text-gray-400 leading-relaxed">
          {price.executive_summary}
        </p>
      </div>

      {/* Top Exposures */}
      {price.top_exposures && price.top_exposures.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <p className="text-sm font-medium text-gray-300 mb-3">
            Top Exposures
          </p>
          <div className="space-y-3">
            {price.top_exposures.map((exp, i) => (
              <div
                key={i}
                className="flex justify-between items-start gap-4"
              >
                <div className="flex-1">
                  <p className="text-sm text-gray-300">
                    {exp.exposure || exp.name}
                  </p>
                  {exp.mitigation_available && (
                    <p className="text-xs text-gray-600 mt-0.5">
                      Mitigation: {exp.mitigation_available}
                    </p>
                  )}
                </div>
                <span
                  className={`text-xs font-medium px-2 py-0.5 rounded ${
                    exp.severity === "high" || exp.severity === "critical"
                      ? "bg-red-400/10 text-red-400"
                      : exp.severity === "medium"
                      ? "bg-amber-400/10 text-amber-400"
                      : "bg-gray-800 text-gray-400"
                  }`}
                >
                  {exp.severity}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk Scenarios */}
      {price.scenarios && price.scenarios.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <p className="text-sm font-medium text-gray-300 mb-3">
            Risk Scenarios
          </p>
          <div className="space-y-4">
            {price.scenarios.map((s, i) => (
              <div
                key={i}
                className="border-l-2 border-amber-500/50 pl-3 space-y-1"
              >
                <p className="text-sm font-medium text-gray-300">
                  {s.scenario_type}
                </p>
                <div className="flex gap-3 text-xs text-gray-500">
                  <span>Probability: {s.probability}</span>
                  <span>Severity: {s.severity}</span>
                </div>
                <p className="text-xs text-amber-400/70">
                  Expected loss: {s.expected_loss_range}
                </p>
                {s.applicable_doctrines && s.applicable_doctrines.length > 0 && (
                  <p className="text-xs text-gray-600">
                    Legal basis: {s.applicable_doctrines.join(", ")}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {price.recommendations && price.recommendations.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <p className="text-sm font-medium text-gray-300 mb-3">
            Recommendations
          </p>
          <div className="space-y-3">
            {price.recommendations.slice(0, 5).map((r, i) => (
              <div key={i} className="flex items-start gap-3">
                <span
                  className={`text-xs font-medium px-1.5 py-0.5 rounded mt-0.5 ${priorityColor(
                    r.priority
                  )} bg-gray-800`}
                >
                  {r.priority}
                </span>
                <div className="flex-1">
                  <p className="text-sm text-gray-300">{r.action}</p>
                  {r.reasoning && (
                    <p className="text-xs text-gray-600 mt-0.5">
                      {r.reasoning}
                    </p>
                  )}
                  {r.impact && (
                    <p className="text-xs text-amber-400/60 mt-0.5">
                      Impact: {r.impact}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Gaps */}
      {price.data_gaps && price.data_gaps.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
          <p className="text-sm font-medium text-gray-300 mb-2">
            Information Gaps
          </p>
          <p className="text-xs text-gray-500 mb-2">
            These gaps reduce assessment confidence. Providing this information
            would improve accuracy.
          </p>
          <div className="space-y-1">
            {price.data_gaps.map((gap, i) => (
              <p key={i} className="text-sm text-gray-400">
                &bull; {gap}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
