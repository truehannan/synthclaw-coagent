import { useEffect, useState } from "react";
import { system, models, society } from "@/lib/api";
import { Cpu, HardDrive, MemoryStick, Clock, Brain, Activity, BarChart3, Terminal } from "lucide-react";

interface UsageEntry {
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  calls: number;
}

export default function Dashboard() {
  const [status, setStatus] = useState<any>(null);
  const [model, setModel] = useState("");
  const [societyStatus, setSocietyStatus] = useState<any>(null);
  const [usage, setUsage] = useState<UsageEntry[]>([]);
  const [cmd, setCmd] = useState("");
  const [cmdOutput, setCmdOutput] = useState("");
  const [cmdRunning, setCmdRunning] = useState(false);

  useEffect(() => {
    system.status().then(setStatus).catch(() => {});
    models.current().then((r) => setModel(r.model)).catch(() => {});
    society.status().then(setSocietyStatus).catch(() => {});
    models.usage().then((r) => setUsage(r.usage || [])).catch(() => {});
  }, []);

  const formatBytes = (b: number) => {
    if (!b) return "0";
    const gb = b / 1073741824;
    return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(b / 1048576).toFixed(0)} MB`;
  };

  const formatUptime = (s: number) => {
    if (!s) return "—";
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    return d > 0 ? `${d}d ${h}h` : h > 0 ? `${h}h ${m}m` : `${m}m`;
  };

  const formatTokens = (n: number) => {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
  };

  // Simple cost estimation (approximate $/1M tokens)
  const estimateCost = (entry: UsageEntry): string => {
    // Default pricing tiers (rough averages)
    const inputRate = 0.5; // $/1M input tokens
    const outputRate = 1.5; // $/1M output tokens
    const cost = (entry.input_tokens * inputRate + entry.output_tokens * outputRate) / 1_000_000;
    if (cost < 0.001) return "<$0.001";
    return `~$${cost.toFixed(3)}`;
  };

  const totalTokens = usage.reduce((sum, u) => sum + u.total_tokens, 0);
  const totalCalls = usage.reduce((sum, u) => sum + u.calls, 0);

  async function runCommand() {
    if (!cmd.trim() || cmdRunning) return;
    setCmdRunning(true);
    setCmdOutput("");
    try {
      const res = await system.run(cmd.trim());
      const out = (res.stdout || "") + (res.stderr ? `\n${res.stderr}` : "");
      setCmdOutput(out || "(no output)");
    } catch (err: any) {
      setCmdOutput(`Error: ${err.message}`);
    } finally {
      setCmdRunning(false);
    }
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-lg font-bold">[+] Dashboard</h1>
        <p className="text-xs text-muted">System overview and agent status</p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard icon={Cpu} label="CPU" value={status ? `${status.cpu_percent}%` : "—"} />
        <StatCard icon={MemoryStick} label="Memory" value={status ? `${status.memory.percent}%` : "—"} sub={status ? formatBytes(status.memory.used) : ""} />
        <StatCard icon={HardDrive} label="Disk" value={status ? `${status.disk.percent}%` : "—"} sub={status ? formatBytes(status.disk.used) : ""} />
        <StatCard icon={Clock} label="Uptime" value={status ? formatUptime(status.uptime) : "—"} />
      </div>

      {/* Model & Society */}
      <div className="mt-6 grid gap-3 md:grid-cols-2">
        <div className="rounded-sm border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-xs text-muted">
            <Activity className="h-3.5 w-3.5" />
            Active Model
          </div>
          <p className="mt-2 text-sm font-semibold text-primary">{model || "Not configured"}</p>
        </div>

        <div className="rounded-sm border border-border bg-card p-4">
          <div className="flex items-center gap-2 text-xs text-muted">
            <Brain className="h-3.5 w-3.5" />
            Agent Society
          </div>
          <p className="mt-2 text-sm font-semibold">
            {societyStatus?.agents?.active || 0} active agents
          </p>
          <p className="text-xs text-muted">
            {societyStatus?.agents?.total_completed || 0} tasks completed
          </p>
        </div>
      </div>

      {/* Usage section */}
      <div className="mt-6 rounded-sm border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-xs text-muted">
            <BarChart3 className="h-3.5 w-3.5" />
            Token Usage
          </div>
          <div className="flex gap-4 text-[10px] text-muted">
            <span>Total: <span className="text-foreground font-medium">{formatTokens(totalTokens)}</span> tokens</span>
            <span>Calls: <span className="text-foreground font-medium">{totalCalls}</span></span>
          </div>
        </div>

        {usage.length > 0 ? (
          <div className="space-y-2">
            {/* Header */}
            <div className="grid grid-cols-5 gap-2 text-[10px] text-muted border-b border-border/50 pb-1">
              <span className="col-span-2">Model</span>
              <span className="text-right">Input</span>
              <span className="text-right">Output</span>
              <span className="text-right">Est. Cost</span>
            </div>
            {/* Rows */}
            {usage.map((u) => (
              <div key={u.model} className="grid grid-cols-5 gap-2 text-xs items-center">
                <span className="col-span-2 text-foreground font-medium truncate" title={u.model}>
                  {u.model}
                </span>
                <span className="text-right text-muted">{formatTokens(u.input_tokens)}</span>
                <span className="text-right text-muted">{formatTokens(u.output_tokens)}</span>
                <span className="text-right text-primary font-medium">{estimateCost(u)}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted text-center py-4">No usage data yet. Start chatting to track tokens.</p>
        )}
      </div>

      {/* Quick info */}
      {status && (
        <div className="mt-6 rounded-sm border border-border bg-card p-4">
          <p className="text-xs text-muted">System</p>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs md:grid-cols-4">
            <div><span className="text-muted">Host:</span> <span className="text-foreground">{status.hostname}</span></div>
            <div><span className="text-muted">OS:</span> <span className="text-foreground">{status.platform}</span></div>
            <div><span className="text-muted">Python:</span> <span className="text-foreground">{status.python}</span></div>
            <div><span className="text-muted">RAM:</span> <span className="text-foreground">{formatBytes(status.memory.total)}</span></div>
          </div>
        </div>
      )}

      {/* Run Command */}
      <div className="mt-6 rounded-sm border border-border bg-card p-4">
        <div className="flex items-center gap-2 text-xs text-muted mb-3">
          <Terminal className="h-3.5 w-3.5" />
          Run Command
        </div>
        <div className="flex gap-2">
          <input
            value={cmd}
            onChange={(e) => setCmd(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runCommand()}
            placeholder="ls -la, systemctl status, etc..."
            className="flex-1 rounded-sm border border-border bg-background px-3 py-2 text-xs font-mono text-foreground placeholder-muted/50 outline-none focus:border-primary"
          />
          <button
            onClick={runCommand}
            disabled={!cmd.trim() || cmdRunning}
            className="rounded-sm bg-primary px-3 py-2 text-xs text-white hover:bg-primary-hover disabled:opacity-30"
          >
            {cmdRunning ? "..." : "Run"}
          </button>
        </div>
        {cmdOutput && (
          <pre className="mt-3 max-h-48 overflow-y-auto rounded-sm bg-background p-3 text-[10px] leading-relaxed text-muted whitespace-pre-wrap font-mono">
            {cmdOutput}
          </pre>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, sub }: { icon: any; label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-sm border border-border bg-card p-4">
      <div className="flex items-center gap-2 text-xs text-muted">
        <Icon className="h-3.5 w-3.5" />
        {label}
      </div>
      <p className="mt-1 text-xl font-bold">{value}</p>
      {sub && <p className="text-[10px] text-muted">{sub}</p>}
    </div>
  );
}
