"use client";

import { cn } from "@/lib/utils";
import { ConfusionCell } from "@/lib/api";

interface ConfusionMatrixProps {
  cells: ConfusionCell[];
  total: number;
}

const ACTUAL_LABELS = ["benign", "replay", "injection", "face_swap", "doc_tamper"];
const PREDICTED_LABELS = ["pass", "review", "fail"];

export function ConfusionMatrix({ cells, total }: ConfusionMatrixProps) {
  // Create a map for quick lookup
  const cellMap = new Map<string, number>();
  cells.forEach((cell) => {
    cellMap.set(`${cell.actual}-${cell.predicted}`, cell.count);
  });

  const getCount = (actual: string, predicted: string) => {
    return cellMap.get(`${actual}-${predicted}`) || 0;
  };

  const getColor = (actual: string, predicted: string, count: number) => {
    if (count === 0) return "bg-muted";

    // For benign: pass is correct (green), others are false positives (yellow/red)
    if (actual === "benign") {
      if (predicted === "pass") return "bg-success/40";
      if (predicted === "review") return "bg-warning/40";
      return "bg-danger/40";
    }

    // For attacks: fail is best (green), review is ok (yellow), pass is bad (red)
    if (predicted === "fail") return "bg-success/40";
    if (predicted === "review") return "bg-warning/40";
    return "bg-danger/40";
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="p-2 text-left text-muted-foreground">Actual ↓</th>
            {PREDICTED_LABELS.map((label) => (
              <th
                key={label}
                className="p-2 text-center text-muted-foreground capitalize"
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ACTUAL_LABELS.map((actual) => (
            <tr key={actual}>
              <td className="p-2 font-medium capitalize">
                {actual.replace("_", " ")}
              </td>
              {PREDICTED_LABELS.map((predicted) => {
                const count = getCount(actual, predicted);
                const percentage = total > 0 ? (count / total) * 100 : 0;
                return (
                  <td key={predicted} className="p-1">
                    <div
                      className={cn(
                        "p-3 rounded text-center transition-colors",
                        getColor(actual, predicted, count)
                      )}
                    >
                      <div className="font-semibold">{count}</div>
                      {count > 0 && (
                        <div className="text-xs text-muted-foreground">
                          {percentage.toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-4 flex gap-4 text-xs text-muted-foreground justify-center">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-success/40" />
          <span>Correct</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-warning/40" />
          <span>Review needed</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-danger/40" />
          <span>Incorrect</span>
        </div>
      </div>
    </div>
  );
}






