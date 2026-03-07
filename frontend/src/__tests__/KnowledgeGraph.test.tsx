import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import KnowledgeGraph from "@/components/KnowledgeGraph";

const mockGraphData = {
  doctrines: [
    { name: "apparent_authority" },
    { name: "negligent_misrepresentation" },
  ],
  regulations: [
    { short_name: "EU AI Act" },
    { short_name: "GDPR" },
  ],
  risk_factors: [
    { name: "autonomy_level", weight: 0.8 },
    { name: "hallucination_risk", weight: 0.6 },
  ],
  mitigations: [
    { name: "human_in_loop", effectiveness: 0.7 },
    { name: "output_filtering", effectiveness: 0.5 },
  ],
  doctrine_edges: [
    {
      source: "apparent_authority",
      target: "negligent_misrepresentation",
      relationship: "related_to",
    },
  ],
  mitigation_edges: [
    {
      source: "human_in_loop",
      target: "autonomy_level",
      reduction: 0.4,
    },
    {
      source: "output_filtering",
      target: "hallucination_risk",
      reduction: 0.3,
    },
  ],
};

describe("KnowledgeGraph", () => {
  it("renders without crashing", () => {
    // ForceGraph2D is mocked via next/dynamic mock in setup
    const { container } = render(<KnowledgeGraph data={mockGraphData} />);
    expect(container).toBeTruthy();
  });

  it("renders with empty data arrays", () => {
    const emptyData = {
      doctrines: [],
      regulations: [],
      risk_factors: [],
      mitigations: [],
      doctrine_edges: [],
      mitigation_edges: [],
    };
    const { container } = render(<KnowledgeGraph data={emptyData} />);
    expect(container).toBeTruthy();
  });

  it("handles edges referencing non-existent nodes", () => {
    const dataWithBadEdges = {
      ...mockGraphData,
      doctrine_edges: [
        {
          source: "nonexistent_doctrine",
          target: "apparent_authority",
          relationship: "test",
        },
      ],
    };
    // Should not throw — invalid edges are filtered
    const { container } = render(
      <KnowledgeGraph data={dataWithBadEdges} />
    );
    expect(container).toBeTruthy();
  });
});
