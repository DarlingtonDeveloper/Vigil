import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the anthropic provider before importing the route
vi.mock("@ai-sdk/anthropic", () => ({
  anthropic: vi.fn(() => ({
    specificationVersion: "v3",
    provider: "anthropic",
    modelId: "claude-sonnet-4-20250514",
    doGenerate: vi.fn(),
    doStream: vi.fn(),
  })),
}));

// Mock streamText to avoid real LLM calls
const mockToUIMessageStreamResponse = vi.fn(() => new Response("stream", { status: 200 }));

vi.mock("ai", async (importOriginal) => {
  const actual = await importOriginal<typeof import("ai")>();
  return {
    ...actual,
    streamText: vi.fn(() => ({
      toUIMessageStreamResponse: mockToUIMessageStreamResponse,
    })),
  };
});

import { POST } from "@/app/api/chat/route";
import { streamText } from "ai";

describe("POST /api/chat", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("calls streamText with Anthropic model and system prompt", async () => {
    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: [
          {
            id: "1",
            role: "user",
            parts: [{ type: "text", text: "Assess my deployment" }],
          },
        ],
      }),
    });

    await POST(req);

    expect(streamText).toHaveBeenCalledOnce();
    const call = (streamText as any).mock.calls[0][0];
    expect(call.system).toContain("Vigil");
    expect(call.system).toContain("risk pricing engine");
  });

  it("returns a UIMessageStreamResponse", async () => {
    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    const response = await POST(req);

    expect(mockToUIMessageStreamResponse).toHaveBeenCalled();
    expect(response).toBeInstanceOf(Response);
    expect(response.status).toBe(200);
  });

  it("configures three tools: assessDeployment, queryKnowledge, getMitigations", async () => {
    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    await POST(req);

    const call = (streamText as any).mock.calls[0][0];
    expect(call.tools).toBeDefined();
    expect(call.tools.assessDeployment).toBeDefined();
    expect(call.tools.queryKnowledge).toBeDefined();
    expect(call.tools.getMitigations).toBeDefined();
  });

  it("assessDeployment tool calls backend /api/assess", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          risk_price: { overall_risk_score: 0.5 },
        }),
    });
    globalThis.fetch = mockFetch;

    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    await POST(req);

    const call = (streamText as any).mock.calls[0][0];
    const assessTool = call.tools.assessDeployment;

    const result = await assessTool.execute({
      description: "A customer support bot",
      jurisdictions: ["UK"],
      sector: "finance",
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/api/assess",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          "X-API-Key": "pg-demo-key-2025",
        }),
      })
    );
    expect(result).toEqual({
      risk_price: { overall_risk_score: 0.5 },
    });
  });

  it("assessDeployment tool returns error on failure", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      text: () => Promise.resolve("Internal Server Error"),
    });

    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    await POST(req);

    const call = (streamText as any).mock.calls[0][0];
    const result = await call.tools.assessDeployment.execute({
      description: "test",
      jurisdictions: ["UK"],
    });

    expect(result).toEqual({
      error: "Assessment failed: Internal Server Error",
    });
  });

  it("queryKnowledge tool calls backend /api/knowledge/:type", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ doctrines: [] }),
    });
    globalThis.fetch = mockFetch;

    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    await POST(req);

    const call = (streamText as any).mock.calls[0][0];
    await call.tools.queryKnowledge.execute({
      query_type: "doctrines",
      jurisdiction: "UK",
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/api/knowledge/doctrines?jurisdiction=UK",
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-API-Key": "pg-demo-key-2025",
        }),
      })
    );
  });

  it("getMitigations tool calls backend /api/knowledge/mitigations/:factor", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ mitigations: [] }),
    });
    globalThis.fetch = mockFetch;

    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    await POST(req);

    const call = (streamText as any).mock.calls[0][0];
    await call.tools.getMitigations.execute({
      risk_factor: "hallucination_risk",
    });

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8080/api/knowledge/mitigations/hallucination_risk",
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-API-Key": "pg-demo-key-2025",
        }),
      })
    );
  });

  it("configures stopWhen for multi-step tool calls", async () => {
    const req = new Request("http://localhost:3000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    await POST(req);

    const call = (streamText as any).mock.calls[0][0];
    expect(call.stopWhen).toBeDefined();
  });
});
