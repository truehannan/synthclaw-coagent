import { useEffect, useState } from "react";
import { society as api } from "@/lib/api";
import { RefreshCw, Trash2 } from "lucide-react";

const ROLE_COLORS: Record<string, string> = {
  orchestrator: "#e85d04",
  researcher: "#4dabf7",
  executor: "#51cf66",
  reviewer: "#fcc419",
  observer: "#cc5de8",
  specialist: "#ff6b6b",
};

export default function Society() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      const res = await api.status();
      setStatus(res);
    } catch {}
  }

  async function reset() {
    await api.reset();
    load();
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-bold">[+] Agent Society</h1>
          <p className="text-xs text-muted">Multi-agent orchestration status</p>
        </div>
        <div className="flex gap-2">
          <button onClick={load} className="rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-primary hover:text-primary">
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          <button onClick={reset} className="rounded-sm border border-border px-3 py-1.5 text-xs text-muted hover:border-danger hover:text-danger">
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Summary */}
      {status?.agents && (
        <div className="mb-6 grid grid-cols-3 gap-3">
          <div className="rounded-sm border border-border bg-card p-4 text-center">
            <p className="text-2xl font-bold">{status.agents.active || 0}</p>
            <p className="text-[10px] text-muted">Active</p>
          </div>
          <div className="rounded-sm border border-border bg-card p-4 text-center">
            <p className="text-2xl font-bold">{status.agents.total_completed || 0}</p>
            <p className="text-[10px] text-muted">Completed</p>
          </div>
          <div className="rounded-sm border border-border bg-card p-4 text-center">
            <p className="text-2xl font-bold">{Object.keys(status.agents.by_role || {}).length}</p>
            <p className="text-[10px] text-muted">Roles Active</p>
          </div>
        </div>
      )}

      {/* Agent tree */}
      {status?.tree && status.tree.length > 0 ? (
        <div className="rounded-sm border border-border bg-card p-4">
          <p className="mb-3 text-xs text-muted">Agent Hierarchy</p>
          {status.tree.map((node: any) => (
            <AgentTreeNode key={node.id} node={node} depth={0} />
          ))}
        </div>
      ) : (
        <div className="rounded-sm border border-border bg-card p-8 text-center">
          <p className="text-sm text-muted">No active agents</p>
          <p className="mt-1 text-[10px] text-muted">
            Send a complex task and agents will spawn automatically
          </p>
        </div>
      )}

      {/* Active agents list */}
      {status?.active && status.active.length > 0 && (
        <div className="mt-6 space-y-1">
          <p className="mb-2 text-xs text-muted">Active Agents</p>
          {status.active.map((a: any) => (
            <div key={a.id} className="flex items-center gap-3 rounded-sm border border-border bg-card px-4 py-2.5">
              <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: ROLE_COLORS[a.role] || "#888" }} />
              <span className="text-xs font-semibold">{a.name}</span>
              <span className="text-[10px] text-muted">[{a.status}]</span>
              <span className="flex-1 truncate text-[10px] text-muted">{a.task}</span>
              <span className="text-[10px] text-muted">{a.elapsed}s</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function AgentTreeNode({ node, depth }: { node: any; depth: number }) {
  const indent = depth * 16;
  const color = ROLE_COLORS[node.role] || "#888";
  return (
    <div>
      <div className="flex items-center gap-2 py-1" style={{ paddingLeft: `${indent}px` }}>
        {depth > 0 && <span className="text-border">├─</span>}
        <div className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-xs font-semibold">{node.name || node.role}</span>
        <span className="text-[10px] text-muted">[{node.status}]</span>
      </div>
      {node.children?.map((child: any) => (
        <AgentTreeNode key={child.id} node={child} depth={depth + 1} />
      ))}
    </div>
  );
}
