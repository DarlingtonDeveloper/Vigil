"use client";

import Chat from "@/components/Chat";
import RiskDashboard from "@/components/RiskDashboard";
import { useState } from "react";

export default function Home() {
  const [assessmentData, setAssessmentData] = useState<Record<string, unknown> | null>(null);

  return (
    <main className="flex h-screen bg-gray-950 text-gray-100">
      {/* Left panel — Chat */}
      <div className="w-1/2 flex flex-col border-r border-gray-800">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-xl font-bold tracking-tight">
            <span className="text-amber-400">Fault</span>Line
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            AI agent deployment risk pricing
          </p>
        </div>
        <Chat onAssessment={setAssessmentData} />
      </div>

      {/* Right panel — Risk Dashboard */}
      <div className="w-1/2 flex flex-col overflow-y-auto">
        <div className="p-4 border-b border-gray-800">
          <h2 className="text-sm font-medium text-gray-400">
            Risk Assessment
          </h2>
        </div>
        {assessmentData ? (
          <RiskDashboard data={assessmentData} />
        ) : (
          <div className="flex items-center justify-center h-full text-gray-600">
            <div className="text-center space-y-3 max-w-md px-8">
              <p className="text-lg">Describe an agentic deployment to assess its risk</p>
              <p className="text-sm text-gray-700">
                e.g., &quot;We&apos;re deploying a customer-facing AI agent that can access our CRM,
                send emails on behalf of account managers, and process refunds up to &pound;500.
                It operates in the UK and EU. We have a general operator reviewing flagged interactions.&quot;
              </p>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
