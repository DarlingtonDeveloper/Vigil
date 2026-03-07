import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import Home from "@/app/page";

// Mock useChat
vi.mock("@ai-sdk/react", () => ({
  useChat: () => ({
    messages: [],
    sendMessage: vi.fn(),
    status: "ready",
  }),
}));

vi.mock("ai", () => ({
  DefaultChatTransport: class MockTransport {
    constructor() {}
  },
}));

describe("Home page", () => {
  it("renders FaultLine branding", () => {
    render(<Home />);
    expect(screen.getByText("Fault")).toBeInTheDocument();
    expect(screen.getByText("Line")).toBeInTheDocument();
    expect(
      screen.getByText("AI agent deployment risk pricing")
    ).toBeInTheDocument();
  });

  it("renders split-screen layout with chat and dashboard panels", () => {
    render(<Home />);
    expect(screen.getByText("Risk Assessment")).toBeInTheDocument();
    expect(
      screen.getByText("Describe an agentic deployment to assess its risk")
    ).toBeInTheDocument();
  });

  it("shows placeholder text in dashboard when no assessment data", () => {
    render(<Home />);
    expect(
      screen.getByText(/deploying a customer-facing AI agent/)
    ).toBeInTheDocument();
  });

  it("renders the chat input", () => {
    render(<Home />);
    expect(
      screen.getByPlaceholderText("Describe your agentic deployment...")
    ).toBeInTheDocument();
  });
});
