import { useEffect, useState } from "react";
import { system, models, society } from "@/lib/api";
import { Cpu, HardDrive, MemoryStick, Clock, Brain, Activity } from "lucide-react";

export default function Dashboard() {
  const [status, setStatus] = useState<any>(null);
  const [model, setModel] = useState("");
  const [societyStatus, setSocietyStatus] = useState<any>(null);

  useEffect(() => {
    system.status().then(setStatus).catch(() => {});
    models.current().then((r) => setModel(r.model)).catch(() => {});
    society.status().then(setSocietyStatus).catch(() => {});
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
