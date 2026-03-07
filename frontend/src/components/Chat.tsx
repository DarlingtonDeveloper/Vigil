"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useRef, useState, useMemo } from "react";

interface ChatProps {
  onAssessment: (data: Record<string, unknown>) => void;
}

export default function Chat({ onAssessment }: ChatProps) {
  const [input, setInput] = useState("");
  const transport = useMemo(() => new DefaultChatTransport({ api: "/api/chat" }), []);

  const { messages, sendMessage, status } = useChat({
    transport,
    onFinish: ({ message }) => {
      // Check message parts for assessment tool results
      try {
        for (const part of message.parts) {
          if (
            part.type.startsWith("tool-") &&
            "toolName" in part &&
            part.toolName === "assessDeployment" &&
            "state" in part &&
            part.state === "output-available" &&
            "output" in part &&
            part.output &&
            typeof part.output === "object" &&
            !("error" in (part.output as Record<string, unknown>))
          ) {
            onAssessment(part.output as Record<string, unknown>);
          }
        }
      } catch (e) {
        console.error("Failed to extract assessment:", e);
      }
    },
  });

  const isLoading = status === "submitted" || status === "streaming";
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const suggestedPrompts = [
    "We're deploying a customer support agent that can access our CRM, send emails to customers, and issue refunds up to \u00A3500. It uses Claude and operates in the UK.",
    "We have an internal research agent that summarises documents and answers questions from our knowledge base. No external access, internal only.",
    "We're building a financial advisor chatbot that provides investment recommendations to retail customers. It can access portfolio data and execute trades. UK and EU regulated.",
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const text = input;
    setInput("");
    sendMessage({ text });
  };

  const handlePromptClick = (prompt: string) => {
    setInput("");
    sendMessage({ text: prompt });
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="space-y-4 mt-4">
            <p className="text-sm text-gray-500">Try one of these:</p>
            {suggestedPrompts.map((prompt, i) => (
              <button
                key={i}
                onClick={() => handlePromptClick(prompt)}
                className="block w-full text-left p-3 rounded-lg bg-gray-900 border border-gray-800 text-sm text-gray-400 hover:border-amber-500/50 hover:text-gray-300 transition-colors"
              >
                {prompt}
              </button>
            ))}
          </div>
        )}

        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex ${
              m.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-[85%] rounded-lg px-4 py-3 text-sm leading-relaxed ${
                m.role === "user"
                  ? "bg-amber-600 text-white"
                  : "bg-gray-800 text-gray-200"
              }`}
            >
              {m.parts.map((part, i) => {
                if (part.type === "text") {
                  return <span key={i}>{part.text}</span>;
                }
                if (
                  part.type.startsWith("tool-") &&
                  "toolName" in part &&
                  "state" in part
                ) {
                  return (
                    <div
                      key={i}
                      className="mb-2 p-2 rounded bg-gray-700/50 text-xs text-gray-400"
                    >
                      {(part.state === "input-available" ||
                        part.state === "input-streaming") && (
                        <span className="animate-pulse">
                          Running {String(part.toolName)}...
                        </span>
                      )}
                      {part.state === "output-available" && (
                        <span className="text-green-400">
                          &#10003; {String(part.toolName)} complete
                        </span>
                      )}
                      {part.state === "output-error" && (
                        <span className="text-red-400">
                          &#10007; {String(part.toolName)}:{" "}
                          {"errorText" in part
                            ? String(part.errorText)
                            : "Error"}
                        </span>
                      )}
                    </div>
                  );
                }
                return null;
              })}
            </div>
          </div>
        ))}

        {isLoading && messages.length > 0 && messages[messages.length - 1].role === "user" && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-lg px-4 py-3 text-sm text-gray-400">
              <span className="animate-pulse">Analyzing deployment risk...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your agentic deployment..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-amber-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-amber-600 hover:bg-amber-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg px-5 py-2.5 text-sm font-medium transition-colors"
          >
            Assess
          </button>
        </div>
      </form>
    </div>
  );
}
