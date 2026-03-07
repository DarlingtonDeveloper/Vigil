import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vigil — AI Agent Deployment Risk Pricing",
  description:
    "Price the legal and operational risk of deploying AI agents",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
