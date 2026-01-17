import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Sidebar } from "@/components/layout/sidebar";
import { Navbar } from "@/components/layout/navbar";
import { ConnectionStatus } from "@/components/layout/connection-status";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
});

export const metadata: Metadata = {
  title: "KYC Sentinel Lab",
  description:
    "Red-team your remote identity flow with safe, synthetic attacks and explainable detection",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`} suppressHydrationWarning>
      <body className={`${inter.className} min-h-screen antialiased`}>
        <Providers>
          <ConnectionStatus />
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <div className="flex flex-1 flex-col overflow-hidden">
              <Navbar />
              <main className="flex-1 overflow-auto gradient-mesh">
                <div className="container py-4 md:py-6 pb-20 lg:pb-6">{children}</div>
              </main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}



