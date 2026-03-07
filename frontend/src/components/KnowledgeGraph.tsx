"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

interface GraphNode {
  id: string;
  name: string;
  type: string;
  color: string;
  val: number;
  x?: number;
  y?: number;
}

interface GraphLink {
  source: string;
  target: string;
  label: string;
}

interface KnowledgeGraphProps {
  data: {
    doctrines: Array<{ name: string }>;
    regulations: Array<{ short_name: string }>;
    risk_factors: Array<{ name: string; weight?: number }>;
    mitigations: Array<{ name: string; effectiveness?: number }>;
    doctrine_edges: Array<{ source: string; target: string; relationship: string }>;
    mitigation_edges: Array<{ source: string; target: string; reduction?: number }>;
  };
}

const TYPE_COLORS: Record<string, string> = {
  doctrine: "#ef4444",
  regulation: "#8b5cf6",
  risk_factor: "#f59e0b",
  mitigation: "#10b981",
};

export default function KnowledgeGraph({ data }: KnowledgeGraphProps) {
  const graphData = useMemo(() => {
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];

    (data.doctrines || []).forEach((d) => {
      nodes.push({
        id: `doctrine:${d.name}`,
        name: d.name,
        type: "doctrine",
        color: TYPE_COLORS.doctrine,
        val: 4,
      });
    });

    (data.regulations || []).forEach((r) => {
      nodes.push({
        id: `regulation:${r.short_name}`,
        name: r.short_name,
        type: "regulation",
        color: TYPE_COLORS.regulation,
        val: 5,
      });
    });

    (data.risk_factors || []).forEach((rf) => {
      nodes.push({
        id: `risk_factor:${rf.name}`,
        name: rf.name,
        type: "risk_factor",
        color: TYPE_COLORS.risk_factor,
        val: 2 + (rf.weight || 0) * 6,
      });
    });

    (data.mitigations || []).forEach((m) => {
      nodes.push({
        id: `mitigation:${m.name}`,
        name: m.name,
        type: "mitigation",
        color: TYPE_COLORS.mitigation,
        val: 2 + (m.effectiveness || 0) * 5,
      });
    });

    (data.doctrine_edges || []).forEach((e) => {
      links.push({
        source: `doctrine:${e.source}`,
        target: `doctrine:${e.target}`,
        label: e.relationship,
      });
    });

    (data.mitigation_edges || []).forEach((e) => {
      links.push({
        source: `mitigation:${e.source}`,
        target: `risk_factor:${e.target}`,
        label: `reduces ${((e.reduction || 0) * 100).toFixed(0)}%`,
      });
    });

    const nodeIds = new Set(nodes.map((n) => n.id));
    const validLinks = links.filter(
      (l) => nodeIds.has(l.source) && nodeIds.has(l.target)
    );

    return { nodes, links: validLinks };
  }, [data]);

  return (
    <ForceGraph2D
      graphData={graphData}
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      nodeLabel={(node: any) => `${node.name} (${node.type})`}
      nodeColor={(node: any) => node.color}
      nodeVal={(node: any) => node.val}
      linkColor={() => "rgba(255,255,255,0.1)"}
      linkWidth={0.5}
      linkDirectionalArrowLength={3}
      linkDirectionalArrowRelPos={0.8}
      backgroundColor="#030712"
      nodeCanvasObjectMode={() => "after"}
      nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D) => {
        const label = (node.name as string).replace(/_/g, " ");
        const fontSize = 8;
        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "rgba(255,255,255,0.6)";
        ctx.fillText(
          label.length > 20 ? label.substring(0, 20) + "..." : label,
          node.x ?? 0,
          (node.y ?? 0) + (node.val as number) + 6
        );
      }}
      width={typeof window !== "undefined" ? window.innerWidth / 2 : 600}
      height={typeof window !== "undefined" ? window.innerHeight - 60 : 600}
    />
  );
}
