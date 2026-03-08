"use client";

import Chat from "@/components/Chat";

export default function Home() {
  return (
    <div className="flex h-screen bg-void overflow-hidden">
      <div className="relative w-full max-w-[800px] mx-auto flex flex-col">
        <header className="flex items-baseline gap-3 px-8 py-5 border-b border-edge/60">
          <h1
            className="text-[24px] italic tracking-tight text-gold"
            style={{ fontFamily: "var(--font-serif, 'Instrument Serif'), serif" }}
          >
            Vigil
          </h1>
          <span className="text-[11px] uppercase tracking-[0.2em] text-muted font-medium">
            Risk Engine
          </span>
        </header>
        <Chat />
      </div>
    </div>
  );
}
