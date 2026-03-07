import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import RiskDashboard from "@/components/RiskDashboard";

const mockAssessmentData = {
  risk_price: {
    overall_risk_score: 0.72,
    premium_band: "£15,000–£25,000/year",
    confidence: 0.85,
    technical_risk: 0.68,
    legal_exposure: 0.78,
    mitigation_effectiveness: 0.35,
    executive_summary:
      "High-risk deployment due to customer-facing autonomous actions with financial impact.",
    top_exposures: [
      {
        exposure: "Unauthorized financial commitments",
        severity: "high",
        mitigation_available: "Transaction limits with human approval",
      },
      {
        exposure: "Data protection violations",
        severity: "critical",
        mitigation_available: "PII filtering layer",
      },
      {
        name: "Reputational harm",
        severity: "medium",
      },
    ],
    scenarios: [
      {
        scenario_type: "Agent issues unauthorized refund",
        probability: "medium",
        severity: "high",
        expected_loss_range: "£10,000–£50,000",
        applicable_doctrines: ["apparent authority", "negligent misrepresentation"],
      },
      {
        scenario_type: "PII disclosure via hallucination",
        probability: "low",
        severity: "critical",
        expected_loss_range: "£50,000–£500,000",
        applicable_doctrines: ["GDPR Article 82"],
      },
    ],
    recommendations: [
      {
        priority: "critical",
        action: "Implement human-in-the-loop for refunds over £100",
        reasoning: "Reduces apparent authority exposure",
        impact: "40% risk reduction",
      },
      {
        priority: "high",
        action: "Add PII detection and redaction layer",
        reasoning: "Prevents data protection violations",
        impact: "25% risk reduction",
      },
      {
        priority: "medium",
        action: "Implement conversation audit logging",
      },
    ],
    data_gaps: [
      "Volume of daily customer interactions",
      "Existing incident response procedures",
      "Staff training on AI oversight",
    ],
  },
};

describe("RiskDashboard", () => {
  it("returns null when no risk_price data is present", () => {
    const { container } = render(<RiskDashboard data={{}} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders overall risk score", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("Overall Risk Score")).toBeInTheDocument();
    expect(screen.getByText("72")).toBeInTheDocument();
    expect(screen.getByText("out of 100")).toBeInTheDocument();
  });

  it("renders premium band and confidence", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("£15,000–£25,000/year")).toBeInTheDocument();
    expect(screen.getByText("Confidence: 85%")).toBeInTheDocument();
  });

  it("renders score breakdown (technical, legal, mitigation)", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("Technical Risk")).toBeInTheDocument();
    expect(screen.getByText("68%")).toBeInTheDocument();
    expect(screen.getByText("Legal Exposure")).toBeInTheDocument();
    expect(screen.getByText("78%")).toBeInTheDocument();
    expect(screen.getByText("Mitigation Score")).toBeInTheDocument();
    expect(screen.getByText("35%")).toBeInTheDocument();
  });

  it("renders executive summary", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("Executive Summary")).toBeInTheDocument();
    expect(
      screen.getByText(/High-risk deployment due to customer-facing/)
    ).toBeInTheDocument();
  });

  it("renders top exposures with severity badges", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("Top Exposures")).toBeInTheDocument();
    expect(
      screen.getByText("Unauthorized financial commitments")
    ).toBeInTheDocument();
    expect(screen.getByText("Data protection violations")).toBeInTheDocument();
    expect(screen.getByText("Reputational harm")).toBeInTheDocument();

    // Severity badges (may appear in both exposures and scenarios)
    expect(screen.getAllByText("high").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("critical").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("medium").length).toBeGreaterThanOrEqual(1);
  });

  it("renders exposure mitigations when available", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(
      screen.getByText("Mitigation: Transaction limits with human approval")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Mitigation: PII filtering layer")
    ).toBeInTheDocument();
  });

  it("renders risk scenarios with probability, severity, loss range, and legal basis", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("Risk Scenarios")).toBeInTheDocument();
    expect(
      screen.getByText("Agent issues unauthorized refund")
    ).toBeInTheDocument();
    expect(screen.getByText("PII disclosure via hallucination")).toBeInTheDocument();

    // Probability and severity
    expect(screen.getAllByText(/Probability:/).length).toBe(2);
    expect(screen.getAllByText(/Severity:/).length).toBe(2);

    // Expected loss ranges
    expect(screen.getByText("Expected loss: £10,000–£50,000")).toBeInTheDocument();
    expect(
      screen.getByText("Expected loss: £50,000–£500,000")
    ).toBeInTheDocument();

    // Legal basis
    expect(
      screen.getByText(
        "Legal basis: apparent authority, negligent misrepresentation"
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText("Legal basis: GDPR Article 82")
    ).toBeInTheDocument();
  });

  it("renders recommendations with priority, action, reasoning, and impact", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("Recommendations")).toBeInTheDocument();
    expect(
      screen.getByText("Implement human-in-the-loop for refunds over £100")
    ).toBeInTheDocument();
    expect(
      screen.getByText("Reduces apparent authority exposure")
    ).toBeInTheDocument();
    expect(screen.getByText("Impact: 40% risk reduction")).toBeInTheDocument();
    expect(
      screen.getByText("Implement conversation audit logging")
    ).toBeInTheDocument();
  });

  it("renders data gaps section", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    expect(screen.getByText("Information Gaps")).toBeInTheDocument();
    expect(
      screen.getByText(/Volume of daily customer interactions/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Existing incident response procedures/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Staff training on AI oversight/)
    ).toBeInTheDocument();
  });

  it("color codes risk score red for high risk (>60)", () => {
    render(<RiskDashboard data={mockAssessmentData} />);
    const scoreElement = screen.getByText("72");
    expect(scoreElement.className).toContain("text-red-400");
  });

  it("color codes risk score green for low risk (<30)", () => {
    const lowRiskData = {
      risk_price: {
        ...mockAssessmentData.risk_price,
        overall_risk_score: 0.2,
      },
    };
    render(<RiskDashboard data={lowRiskData} />);
    const scoreElement = screen.getByText("20");
    expect(scoreElement.className).toContain("text-green-400");
  });

  it("color codes risk score amber for medium risk (30-60)", () => {
    const medRiskData = {
      risk_price: {
        ...mockAssessmentData.risk_price,
        overall_risk_score: 0.45,
      },
    };
    render(<RiskDashboard data={medRiskData} />);
    const scoreElement = screen.getByText("45");
    expect(scoreElement.className).toContain("text-amber-400");
  });

  it("handles missing optional sections gracefully", () => {
    const minimalData = {
      risk_price: {
        overall_risk_score: 0.5,
        premium_band: "£5,000–£10,000/year",
        confidence: 0.6,
        technical_risk: 0.4,
        legal_exposure: 0.5,
        mitigation_effectiveness: 0.3,
        executive_summary: "Moderate risk deployment.",
      },
    };
    render(<RiskDashboard data={minimalData} />);
    expect(screen.getByText("50")).toBeInTheDocument();
    expect(screen.queryByText("Top Exposures")).not.toBeInTheDocument();
    expect(screen.queryByText("Risk Scenarios")).not.toBeInTheDocument();
    expect(screen.queryByText("Recommendations")).not.toBeInTheDocument();
    expect(screen.queryByText("Information Gaps")).not.toBeInTheDocument();
  });
});
