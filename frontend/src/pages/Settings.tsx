import { useEffect, useState } from "react";
import { system } from "@/lib/api";

export default function Settings() {
  const [cfg, setCfg] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    system.config().then(setCfg).catch(() => {});
    system.logs(30).then((r) => setLogs(r.logs || [])).catch(() => {});
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Settings</h1>
      <p className="mb-6 text-xs text-muted">Configuration and system logs</p>

      {/* Config */}
      {cfg && (
        <div className="mb-6 rounded-sm border border-border bg-card p-4">
          <p className="mb-3 text-xs font-semibold text-muted">Configuration</p>
          <div className="space-y-2 text-xs">
            {Object.entries(cfg).map(([key, value]) => (
              <div key={key} className="flex justify-between border-b border-border/50 pb-1">
                <span className="text-muted">{key}</span>
                <span className="text-foreground font-medium">
                  {typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Logs */}
      <div className="rounded-sm border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-muted">Recent Logs</p>
          <button onClick={() => system.logs(50).then(r => setLogs(r.logs || []))}
            className="text-[10px] text-primary hover:underline">Refresh</button>
        </div>
        <div className="max-h-80 overflow-y-auto rounded-sm bg-background p-3">
          {logs.length > 0 ? (
            <pre className="text-[10px] leading-relaxed text-muted whitespace-pre-wrap">
              {logs.join("\n")}
            </pre>
          ) : (
            <p className="text-xs text-muted text-center py-4">No logs available</p>
          )}
        </div>
      </div>
    </div>
  );
}
