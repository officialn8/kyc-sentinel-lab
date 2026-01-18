"use client";

import { Button } from "@/components/ui/button";
import { EnhancedButton } from "@/components/ui/enhanced-button";
import { Card } from "@/components/ui/card";

export default function TestPage() {
  return (
    <div className="space-y-8 p-8">
      <h1 className="text-2xl font-bold">Button Test Page</h1>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Regular Buttons</h2>
        <div className="flex gap-4">
          <Button onClick={() => alert("Regular button clicked!")}>
            Regular Button
          </Button>
          <Button variant="outline" onClick={() => alert("Outline button clicked!")}>
            Outline Button
          </Button>
          <Button size="lg" onClick={() => alert("Large button clicked!")}>
            Large Button
          </Button>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Enhanced Buttons</h2>
        <div className="flex gap-4">
          <EnhancedButton onClick={() => alert("Enhanced button clicked!")}>
            Enhanced Button
          </EnhancedButton>
          <EnhancedButton variant="outline" onClick={() => alert("Enhanced outline clicked!")}>
            Enhanced Outline
          </EnhancedButton>
          <EnhancedButton size="lg" glow onClick={() => alert("Enhanced glow clicked!")}>
            Enhanced Glow
          </EnhancedButton>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Test Scroll</h2>
        <div className="h-[800px] bg-muted/30 rounded-lg p-4">
          <p>Scroll test area - this should make the page scrollable</p>
          <p className="mt-4">Content at top</p>
          <div className="mt-[700px]">
            <p>Content at bottom</p>
            <Button onClick={() => alert("Bottom button clicked!")}>
              Button at Bottom
            </Button>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <h2 className="text-xl font-semibold">Different Backgrounds</h2>
        <div className="grid grid-cols-3 gap-4">
          <Card className="p-4">
            <Button onClick={() => alert("Button in card!")}>In Card</Button>
          </Card>
          <div className="glass-soft p-4 rounded-lg">
            <Button onClick={() => alert("Button in glass!")}>In Glass</Button>
          </div>
          <div className="gradient-danger p-4 rounded-lg">
            <Button onClick={() => alert("Button in gradient!")}>In Gradient</Button>
          </div>
        </div>
      </div>
    </div>
  );
}