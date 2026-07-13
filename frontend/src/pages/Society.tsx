import { useEffect, useState, useCallback, useRef } from "react";
import { society as api } from "@/lib/api";
import { RefreshCw, Trash2, Play, Pause, Users, Activity, Zap } from "lucide-react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  MarkerType,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

const ROLE_COLORS: Record<string, string> = {
  orchestrator: "#ef4444",
  researcher: "#3b82f6",
  executor: "#10b981",
  reviewer: "#f59e0b",
  observer: "#8b5cf6",
  specialist: "#ec4899",
  planner: "#06b6d4",
  coder: "#22c55e",
  analyst: "#f97316",
};

const ROLE_EMOJI: Record<string, string> = {
  orchestrator: "🎯",
  researcher: "🔍",
  executor: "⚡",
  reviewer: "✅",
  observer: "👁",
  specialist: "🔧",
  planner: "📋",
  coder: "💻",
  analyst: "📊",
};

const STATUS_STYLES: Record<string, { bg: string; text: string; pulse: boolean }> = {
  active: { bg: "#10b98120", text: "#10b981", pulse: true },
  thinking: { bg: "#3b82f620", text: "#3b82f6", pulse: true },
  executing: { bg: "#f59e0b20", text: "#f59e0b", pulse: true },
  waiting: { bg: "#6b728020", text: "#6b7280", pulse: false },
  completed: { bg: "#10b98120", text: "#10b981", pulse: false },
  failed: { bg: "#ef444420", text: "#ef4444", pulse: false },
  idle: { bg: "#6b728020", text: "#6b7280", pulse: false },
};

function buildAgentGraph(data: any): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  if (!data) return { nodes, edges };

  // Build from tree structure
  const tree = data.tree || [];
  const active = data.active || [];

  if (tree.length > 0) {
    // Use tree structure
    let yOffset = 0;
    function addTreeNode(node: any, x: number, y: number, parentId?: string) {
      const color = ROLE_COLORS[node.role] || "#6b7280";
      const emoji = ROLE_EMOJI[node.role] || "●";
      const status = STATUS_STYLES[node.status] || STATUS_STYLES.idle;
      const nodeId = node.id || `agent_${Math.random().toString(36).slice(2)}`;

      nodes.push({
        id: nodeId,
        position: { x, y },
        data: { ...node, emoji },
        style: {
          background: status.bg,
          border: `2px solid ${color}`,
          borderRadius: "8px",
          padding: "12px 16px",
          minWidth: "160px",
          color: "#fafafa",
        },
      });

      if (parentId) {
        edges.push({
          id: `e_${parentId}_${nodeId}`,
          source: parentId,
          target: nodeId,
          animated: status.pulse,
          style: { stroke: color + "80", strokeWidth: 2 },
          markerEnd: { type: MarkerType.ArrowClosed, color },
        });
      }

      (node.children || []).forEach((child: any, i: number) => {
        addTreeNode(child, x + 220, y + i * 120, nodeId);
      });
    }

    tree.forEach((root: any, i: number) => {
      addTreeNode(root, 50, 50 + i * 300);
    });
  } else if (active.length > 0) {
    // No tree — show flat list as simple graph
    // Orchestrator at center, others radiating out
    const orchestrator = active.find((a: any) => a.role === "orchestrator") || active[0];
    const others = active.filter((a: any) => a !== orchestrator);

    const centerX = 300, centerY = 200;
    const orchId = "orch_0";
    const orchColor = ROLE_COLORS[orchestrator.role] || "#ef4444";

    nodes.push({
      id: orchId,
      position: { x: centerX - 80, y: centerY - 30 },
      data: { ...orchestrator, emoji: ROLE_EMOJI[orchestrator.role] || "🎯" },
      style: {
        background: "#ef444420",
        border: `3px solid ${orchColor}`,
        borderRadius: "10px",
        padding: "14px 20px",
        minWidth: "180px",
        color: "#fafafa",
      },
    });

    others.forEach((agent: any, i: number) => {
      const angle = (i / others.length) * 2 * Math.PI - Math.PI / 2;
      const radius = 200;
      const x = centerX + Math.cos(angle) * radius - 70;
      const y = centerY + Math.sin(angle) * radius - 25;
      const color = ROLE_COLORS[agent.role] || "#6b7280";
      const nodeId = `agent_${i}`;
      const status = STATUS_STYLES[agent.status] || STATUS_STYLES.idle;

      nodes.push({
        id: nodeId,
        position: { x, y },
        data: { ...agent, emoji: ROLE_EMOJI[agent.role] || "●" },
        style: {
          background: status.bg,
          border: `2px solid ${color}`,
          borderRadius: "8px",
          padding: "10px 14px",
          minWidth: "140px",
          color: "#fafafa",
        },
      });

      edges.push({
        id: `e_${orchId}_${nodeId}`,
        source: orchId,
        target: nodeId,
        animated: status.pulse,
        style: { stroke: color + "60", strokeWidth: 1.5 },
        markerEnd: { type: MarkerType.ArrowClosed, color: color + "80" },
      });
    });
  }

  return { nodes, edges };
}

export default function Society() {
  const [status, setStatus] = useState<any>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([] as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([] as Edge[]);
  const [streaming, setStreaming] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const streamRef = useRef<EventSource | null>(null);

  useEffect(() => { load(); return () => stopStream(); }, []);

  async function load() {
    try {
      const res = await api.status();
      setStatus(res);
      const { nodes: n, edges: e } = buildAgentGraph(res);
      setNodes(n);
      setEdges(e);
    } catch {}
  }

  async function reset() {
    await api.reset();
    setLogs([]);
    load();
  }

  function startStream() {
    if (streamRef.current) return;
    setStreaming(true);
    // Use polling since SSE needs special handling with auth
    const interval = setInterval(async () => {
      try {
        const res = await api.status();
        setStatus(res);
        const { nodes: n, edges: e } = buildAgentGraph(res);
        setNodes(n);
        setEdges(e);
        // Add to logs
        if (res.active?.length > 0) {
          const newest = res.active[0];
          setLogs(prev => [...prev.slice(-50), `[${new Date().toLocaleTimeString()}] ${newest.name || newest.role}: ${newest.status} — ${newest.task || "..."}`]);
        }
      } catch {}
    }, 1500);
    streamRef.current = interval as any;
  }

  function stopStream() {
    if (streamRef.current) {
      clearInterval(streamRef.current as any);
      streamRef.current = null;
    }
    setStreaming(false);
  }

  // Custom node renderer
  function AgentNodeContent({ data }: { data: any }) {
    const status = STATUS_STYLES[data.status] || STATUS_STYLES.idle;
    return (
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm">{data.emoji}</span>
          <span className="text-[11px] font-bold">{data.name || data.role}</span>
          {status.pulse && <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: status.text }} />}
        </div>
        <p className="text-[9px] opacity-70 truncate">{data.task || data.status || "idle"}</p>
        {data.elapsed && <p className="text-[8px] opacity-50 mt-0.5">{data.elapsed}s</p>}
      </div>
    );
  }

  const totalActive = status?.agents?.active || 0;
  const totalCompleted = status?.agents?.total_completed || 0;

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border bg-card px-4 py-2">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold">[+] Agent Society</h1>
          <div className="flex items-center gap-3 text-[10px] text-muted">
            <span className="flex items-center gap-1"><Users className="h-3 w-3" /> {totalActive} active</span>
            <span className="flex items-center gap-1"><Activity className="h-3 w-3" /> {totalCompleted} done</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {streaming ? (
            <button onClick={stopStream} className="flex items-center gap-1 rounded-sm border border-warning px-2 py-1 text-[10px] text-warning">
              <Pause className="h-3 w-3" /> Stop
            </button>
          ) : (
            <button onClick={startStream} className="flex items-center gap-1 rounded-sm border border-success px-2 py-1 text-[10px] text-success">
              <Play className="h-3 w-3" /> Live
            </button>
          )}
          <button onClick={load} className="rounded-sm border border-border p-1.5 text-muted hover:text-primary">
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          <button onClick={reset} className="rounded-sm border border-border p-1.5 text-muted hover:text-danger">
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Main area: canvas + logs */}
      <div className="flex flex-1 overflow-hidden">
        {/* Canvas */}
        <div className="flex-1">
          {nodes.length > 0 ? (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              fitView
              minZoom={0.3}
              maxZoom={2}
              proOptions={{ hideAttribution: true }}
              nodesDraggable
              nodesConnectable={false}
            >
              <Background color="#1a1a1a" gap={30} />
              <Controls position="bottom-left" style={{ background: "#111", border: "1px solid #333", borderRadius: 4 }} />
              <MiniMap
                nodeColor={(n) => (n.style?.border as string)?.replace(/\dpx solid /, "").replace(/2px solid /, "").replace(/3px solid /, "") || "#666"}
                style={{ background: "#0a0a0a", border: "1px solid #222" }}
              />
              <Panel position="top-right">
                <div className="rounded-sm border border-border bg-card p-3 text-[9px] space-y-1">
                  <p className="font-semibold text-muted mb-1">Roles</p>
                  {Object.entries(ROLE_COLORS).slice(0, 6).map(([role, color]) => (
                    <div key={role} className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
                      <span className="text-muted capitalize">{role}</span>
                    </div>
                  ))}
                </div>
              </Panel>
            </ReactFlow>
          ) : (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <Zap className="h-10 w-10 text-muted mx-auto mb-3" />
                <p className="text-sm text-muted">No active agents</p>
                <p className="text-[10px] text-muted mt-1">Send a complex task in Chat to spawn the agent society</p>
                <p className="text-[9px] text-muted mt-3">
                  Agents: Orchestrator, Researcher, Executor, Reviewer, Observer
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Logs panel */}
        {logs.length > 0 && (
          <div className="w-72 border-l border-border bg-card overflow-hidden flex flex-col">
            <div className="px-3 py-2 border-b border-border flex items-center justify-between">
              <span className="text-[10px] font-semibold text-muted uppercase">Agent Log</span>
              <button onClick={() => setLogs([])} className="text-[9px] text-muted hover:text-foreground">Clear</button>
            </div>
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {logs.map((log, i) => (
                <p key={i} className="text-[9px] text-muted font-mono">{log}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
