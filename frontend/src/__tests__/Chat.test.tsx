import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import Chat from "@/components/Chat";

// Mock @ai-sdk/react useChat hook
const mockSendMessage = vi.fn();
let mockMessages: any[] = [];
let mockStatus = "ready";

vi.mock("@ai-sdk/react", () => ({
  useChat: (options: any) => {
    // Store onFinish for later invocation in tests
    (globalThis as any).__useChatOnFinish = options?.onFinish;
    return {
      messages: mockMessages,
      sendMessage: mockSendMessage,
      status: mockStatus,
    };
  },
}));

vi.mock("ai", () => ({
  DefaultChatTransport: class MockTransport {
    constructor() {}
  },
}));

describe("Chat", () => {
  const mockOnAssessment = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockMessages = [];
    mockStatus = "ready";
  });

  it("renders suggested prompts when there are no messages", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    expect(screen.getByText("Try one of these:")).toBeInTheDocument();
    expect(
      screen.getByText(/customer support agent/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/internal research agent/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/financial advisor chatbot/)
    ).toBeInTheDocument();
  });

  it("renders input field and submit button", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    expect(
      screen.getByPlaceholderText("Describe your agentic deployment...")
    ).toBeInTheDocument();
    expect(screen.getByText("Assess")).toBeInTheDocument();
  });

  it("submit button is disabled when input is empty", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    const button = screen.getByText("Assess");
    expect(button).toBeDisabled();
  });

  it("enables submit button when input has text", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    const input = screen.getByPlaceholderText(
      "Describe your agentic deployment..."
    );
    fireEvent.change(input, { target: { value: "Test deployment" } });
    const button = screen.getByText("Assess");
    expect(button).not.toBeDisabled();
  });

  it("calls sendMessage with text on form submit", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    const input = screen.getByPlaceholderText(
      "Describe your agentic deployment..."
    );
    fireEvent.change(input, { target: { value: "My AI agent" } });
    fireEvent.submit(input.closest("form")!);
    expect(mockSendMessage).toHaveBeenCalledWith({ text: "My AI agent" });
  });

  it("clears input after submit", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    const input = screen.getByPlaceholderText(
      "Describe your agentic deployment..."
    ) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "My AI agent" } });
    fireEvent.submit(input.closest("form")!);
    expect(input.value).toBe("");
  });

  it("calls sendMessage when a suggested prompt is clicked", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    const promptButton = screen.getByText(/customer support agent/);
    fireEvent.click(promptButton);
    expect(mockSendMessage).toHaveBeenCalledWith({
      text: expect.stringContaining("customer support agent"),
    });
  });

  it("renders user messages with amber background", () => {
    mockMessages = [
      {
        id: "1",
        role: "user",
        parts: [{ type: "text", text: "Hello world" }],
      },
    ];
    render(<Chat onAssessment={mockOnAssessment} />);
    const messageEl = screen.getByText("Hello world");
    expect(messageEl.closest("div[class*='bg-amber-600']")).toBeTruthy();
  });

  it("renders assistant messages with gray background", () => {
    mockMessages = [
      {
        id: "2",
        role: "assistant",
        parts: [{ type: "text", text: "Assessment complete" }],
      },
    ];
    render(<Chat onAssessment={mockOnAssessment} />);
    const messageEl = screen.getByText("Assessment complete");
    expect(messageEl.closest("div[class*='bg-gray-800']")).toBeTruthy();
  });

  it("hides suggested prompts when messages exist", () => {
    mockMessages = [
      {
        id: "1",
        role: "user",
        parts: [{ type: "text", text: "Hello" }],
      },
    ];
    render(<Chat onAssessment={mockOnAssessment} />);
    expect(screen.queryByText("Try one of these:")).not.toBeInTheDocument();
  });

  it("renders tool invocation states — running", () => {
    mockMessages = [
      {
        id: "1",
        role: "assistant",
        parts: [
          {
            type: "tool-assessDeployment",
            toolName: "assessDeployment",
            toolCallId: "tc1",
            state: "input-available",
            input: {},
          },
        ],
      },
    ];
    render(<Chat onAssessment={mockOnAssessment} />);
    expect(
      screen.getByText("Running assessDeployment...")
    ).toBeInTheDocument();
  });

  it("renders tool invocation states — complete", () => {
    mockMessages = [
      {
        id: "1",
        role: "assistant",
        parts: [
          {
            type: "tool-assessDeployment",
            toolName: "assessDeployment",
            toolCallId: "tc1",
            state: "output-available",
            input: {},
            output: { risk_price: {} },
          },
        ],
      },
    ];
    render(<Chat onAssessment={mockOnAssessment} />);
    expect(
      screen.getByText(/assessDeployment complete/)
    ).toBeInTheDocument();
  });

  it("renders tool invocation states — error", () => {
    mockMessages = [
      {
        id: "1",
        role: "assistant",
        parts: [
          {
            type: "tool-assessDeployment",
            toolName: "assessDeployment",
            toolCallId: "tc1",
            state: "output-error",
            input: {},
            errorText: "Backend unavailable",
          },
        ],
      },
    ];
    render(<Chat onAssessment={mockOnAssessment} />);
    expect(
      screen.getByText(/assessDeployment.*Backend unavailable/)
    ).toBeInTheDocument();
  });

  it("shows loading indicator when status is submitted and last message is user", () => {
    mockStatus = "submitted";
    mockMessages = [
      {
        id: "1",
        role: "user",
        parts: [{ type: "text", text: "Assess my agent" }],
      },
    ];
    render(<Chat onAssessment={mockOnAssessment} />);
    expect(
      screen.getByText("Analyzing deployment risk...")
    ).toBeInTheDocument();
  });

  it("disables input when loading", () => {
    mockStatus = "streaming";
    render(<Chat onAssessment={mockOnAssessment} />);
    const input = screen.getByPlaceholderText(
      "Describe your agentic deployment..."
    );
    expect(input).toBeDisabled();
  });

  it("does not submit when input is empty", () => {
    render(<Chat onAssessment={mockOnAssessment} />);
    const form = screen.getByPlaceholderText(
      "Describe your agentic deployment..."
    ).closest("form")!;
    fireEvent.submit(form);
    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it("calls onAssessment when onFinish receives assessDeployment result", () => {
    render(<Chat onAssessment={mockOnAssessment} />);

    const onFinish = (globalThis as any).__useChatOnFinish;
    expect(onFinish).toBeDefined();

    const mockResult = { risk_price: { overall_risk_score: 0.7 } };
    onFinish({
      message: {
        id: "msg-1",
        role: "assistant",
        parts: [
          {
            type: "tool-assessDeployment",
            toolName: "assessDeployment",
            toolCallId: "tc1",
            state: "output-available",
            input: {},
            output: mockResult,
          },
          { type: "text", text: "Assessment complete" },
        ],
      },
      messages: [],
      isAbort: false,
      isDisconnect: false,
      isError: false,
    });

    expect(mockOnAssessment).toHaveBeenCalledWith(mockResult);
  });

  it("does not call onAssessment when tool result has error", () => {
    render(<Chat onAssessment={mockOnAssessment} />);

    const onFinish = (globalThis as any).__useChatOnFinish;
    onFinish({
      message: {
        id: "msg-2",
        role: "assistant",
        parts: [
          {
            type: "tool-assessDeployment",
            toolName: "assessDeployment",
            toolCallId: "tc1",
            state: "output-available",
            input: {},
            output: { error: "Assessment failed" },
          },
        ],
      },
      messages: [],
      isAbort: false,
      isDisconnect: false,
      isError: false,
    });

    expect(mockOnAssessment).not.toHaveBeenCalled();
  });
});
