import { useEffect, useState, useCallback, useRef } from "react";
import { memory as api } from "@/lib/api";
import { Plus, Trash2, Edit3, Check, X, Maximize2, List, GitBranch } from "lucide-react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  addEdge,
  Node,
  Edge,
  Connection,
  MarkerType,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

type ViewMode = "canvas" | "list";

// Custom node colors per category
const CATEGORY_COLORS: Record<string, string> = {
  config: "#ef4444",
  mcp: "#8b5cf6",
  user: "#3b82f6",
  session: "#10b981",
  general: "#f59e0b",
  system: "#6366f1",
  agent: "#ec4899",
  default: "#6b7280",
};

function getCategoryColor(key: string): string {
  const sep = key.indexOf("_") > 0 ? key.indexOf("_") : key.indexOf(":") > 0 ? key.indexOf(":") : -1;
  const cat = sep > 0 ? key.slice(0, sep) : "general";
  return CATEGORY_COLORS[cat] || CATEGORY_COLORS.default;
}

function getCategory(key: string): string {
  const sep = key.indexOf("_") > 0 ? key.indexOf("_") : key.indexOf(":") > 0 ? key.indexOf(":") : -1;
  return sep > 0 ? key.slice(0, sep) : "general";
}

function buildGraph(facts: Record<string, string>): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const categories: Record<string, string[]> = {};

  // Group by category
  for (const key of Object.keys(facts)) {
    const cat = getCategory(key);
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(key);
  }

  // Layout: categories in a circle, facts radiating out
  const catKeys = Object.keys(categories);
  const centerX = 400, centerY = 300;
  const catRadius = 250;

  catKeys.forEach((cat, catIdx) => {
    const angle = (catIdx / catKeys.length) * 2 * Math.PI - Math.PI / 2;
    const catX = centerX + Math.cos(angle) * catRadius;
    const catY = centerY + Math.sin(angle) * catRadius;
    const color = CATEGORY_COLORS[cat] || CATEGORY_COLORS.default;

    // Category node
    const catNodeId = `cat_${cat}`;
    nodes.push({
      id: catNodeId,
      position: { x: catX - 50, y: catY - 20 },
      data: { label: cat, count: categories[cat].length },
      type: "default",
      style: {
        background: color + "20",
        border: `2px solid ${color}`,
        borderRadius: "8px",
        padding: "8px 16px",
        fontSize: "11px",
        fontWeight: "bold",
        color: color,
        minWidth: "80px",
        textAlign: "center" as const,
      },
    });

    // Fact nodes radiating from category
    const factCount = categories[cat].length;
    const factRadius = 80 + factCount * 15;
    categories[cat].forEach((key, factIdx) => {
      const factAngle = angle + ((factIdx - factCount / 2) / Math.max(factCount, 1)) * 1.2;
      const fx = catX + Math.cos(factAngle) * factRadius - 60;
      const fy = catY + Math.sin(factAngle) * factRadius - 15;

      const factNodeId = `fact_${key}`;
      const shortKey = key.includes("_") ? key.split("_").slice(1).join("_") : key.includes(":") ? key.split(":").slice(1).join(":") : key;
      const shortVal = (facts[key] || "").slice(0, 25);

      nodes.push({
        id: factNodeId,
        position: { x: fx, y: fy },
        data: { label: `${shortKey}\n${shortVal}${facts[key].length > 25 ? "..." : ""}`, fullKey: key, fullValue: facts[key] },
        type: "default",
        style: {
          background: "#111",
          border: `1px solid ${color}50`,
          borderRadius: "4px",
          padding: "6px 10px",
          fontSize: "9px",
          color: "#ccc",
          maxWidth: "140px",
          whiteSpace: "pre-wrap" as const,
        },
      });

      edges.push({
        id: `e_${catNodeId}_${factNodeId}`,
        source: catNodeId,
        target: factNodeId,
        style: { stroke: color + "40", strokeWidth: 1 },
        markerEnd: { type: MarkerType.ArrowClosed, color: color + "60" },
      });
    });
  });

  return { nodes, edges };
}

export default function Memory() {
  const [facts, setFacts] = useState<Record<string, string>>({});
  const [view, setView] = useState<ViewMode>("canvas");
  const [nodes, setNodes, onNodesChange] = useNodesState([] as Node[]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([] as Edge[]);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [editKey, setEditKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      const res = await api.all();
      const f = res.facts || {};
      setFacts(f);
      if (view === "canvas") {
        const { nodes: n, edges: e } = buildGraph(f);
        setNodes(n);
        setEdges(e);
      }
    } catch {}
  }

  useEffect(() => {
    if (view === "canvas" && Object.keys(facts).length > 0) {
      const { nodes: n, edges: e } = buildGraph(facts);
      setNodes(n);
      setEdges(e);
    }
  }, [view]);

  const onConnect = useCallback((params: Connection) => setEdges(eds => addEdge(params, eds)), [setEdges]);

  function onNodeClick(_: any, node: Node) {
    if (node.id.startsWith("fact_")) {
      setSelectedNode(node);
    }
  }

  async function add() {
    if (!newKey || !newValue) return;
    await api.set(newKey, newValue);
    setNewKey(""); setNewValue("");
    load();
  }

  async function update(key: string) {
    await api.set(key, editValue);
    setEditKey(null);
    setSelectedNode(null);
    load();
  }

  async function remove(key: string) {
    await api.del(key);
    setSelectedNode(null);
    load();
  }

  const totalFacts = Object.keys(facts).length;

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border bg-card px-4 py-2">
        <div>
          <h1 className="text-sm font-bold">[+] Memory</h1>
          <p className="text-[9px] text-muted">{totalFacts} nodes — agent knowledge graph</p>
        </div>
        <div className="flex items-center gap-2">
          {/* Add fact */}
          <input value={newKey} onChange={e => setNewKey(e.target.value)} placeholder="key"
            className="w-24 rounded-sm border border-border bg-background px-2 py-1 text-[10px] outline-none focus:border-primary" />
          <input value={newValue} onChange={e => setNewValue(e.target.value)} placeholder="value"
            onKeyDown={e => e.key === "Enter" && add()}
            className="w-32 rounded-sm border border-border bg-background px-2 py-1 text-[10px] outline-none focus:border-primary" />
          <button onClick={add} disabled={!newKey || !newValue}
            className="rounded-sm bg-primary p-1.5 text-white disabled:opacity-30"><Plus className="h-3 w-3" /></button>
          {/* View toggle */}
          <div className="flex border border-border rounded-sm overflow-hidden ml-2">
            <button onClick={() => setView("canvas")} className={`p-1.5 ${view === "canvas" ? "bg-primary/10 text-primary" : "text-muted"}`}>
              <GitBranch className="h-3 w-3" />
            </button>
            <button onClick={() => setView("list")} className={`p-1.5 ${view === "list" ? "bg-primary/10 text-primary" : "text-muted"}`}>
              <List className="h-3 w-3" />
            </button>
          </div>
        </div>
      </div>

      {/* Canvas View */}
      {view === "canvas" && (
        <div className="flex-1 relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={onNodeClick}
            fitView
            minZoom={0.3}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#222" gap={20} />
            <Controls position="bottom-left" style={{ background: "#111", border: "1px solid #333", borderRadius: 4 }} />
            <MiniMap
              nodeColor={(n) => {
                if (n.id.startsWith("cat_")) return (n.style?.border as string)?.replace("2px solid ", "") || "#666";
                return "#333";
              }}
              style={{ background: "#0a0a0a", border: "1px solid #222" }}
            />
          </ReactFlow>

          {/* Detail panel */}
          {selectedNode && (
            <div className="absolute top-4 right-4 w-64 rounded-sm border border-border bg-card p-4 shadow-lg z-50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-semibold text-primary">{(selectedNode.data as any).fullKey}</span>
                <button onClick={() => setSelectedNode(null)} className="text-muted hover:text-foreground"><X className="h-3 w-3" /></button>
              </div>
              {editKey === (selectedNode.data as any).fullKey ? (
                <div className="space-y-2">
                  <textarea value={editValue} onChange={e => setEditValue(e.target.value)}
                    className="w-full rounded-sm border border-border bg-background px-2 py-1.5 text-[10px] outline-none focus:border-primary resize-none" rows={3} />
                  <div className="flex gap-1">
                    <button onClick={() => update((selectedNode.data as any).fullKey)} className="rounded-sm bg-primary px-2 py-1 text-[9px] text-white"><Check className="h-3 w-3 inline" /> Save</button>
                    <button onClick={() => setEditKey(null)} className="rounded-sm border border-border px-2 py-1 text-[9px] text-muted">Cancel</button>
                  </div>
                </div>
              ) : (
                <>
                  <p className="text-[10px] text-foreground mb-3 whitespace-pre-wrap break-all">{(selectedNode.data as any).fullValue}</p>
                  <div className="flex gap-2">
                    <button onClick={() => { setEditKey((selectedNode.data as any).fullKey); setEditValue((selectedNode.data as any).fullValue); }}
                      className="flex items-center gap-1 text-[9px] text-muted hover:text-primary"><Edit3 className="h-3 w-3" /> Edit</button>
                    <button onClick={() => remove((selectedNode.data as any).fullKey)}
                      className="flex items-center gap-1 text-[9px] text-muted hover:text-danger"><Trash2 className="h-3 w-3" /> Delete</button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* List View */}
      {view === "list" && (
        <div className="flex-1 overflow-y-auto p-4">
          <div className="max-w-3xl mx-auto space-y-1">
            {Object.entries(facts).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5 group">
                <div className="flex-1 min-w-0 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full flex-shrink-0" style={{ background: getCategoryColor(key) }} />
                  <span className="text-xs font-semibold text-primary">{key}</span>
                  <span className="text-xs text-muted">=</span>
                  <span className="text-xs text-foreground truncate">{value}</span>
                </div>
                <div className="flex gap-1 ml-2">
                  <button onClick={() => { setEditKey(key); setEditValue(value); }}
                    className="opacity-0 group-hover:opacity-100 text-muted hover:text-primary"><Edit3 className="h-3.5 w-3.5" /></button>
                  <button onClick={() => remove(key)}
                    className="opacity-0 group-hover:opacity-100 text-muted hover:text-danger"><Trash2 className="h-3.5 w-3.5" /></button>
                </div>
              </div>
            ))}
            {totalFacts === 0 && (
              <div className="text-center py-12">
                <GitBranch className="h-8 w-8 text-muted mx-auto mb-3" />
                <p className="text-xs text-muted">No memories yet</p>
                <p className="text-[10px] text-muted mt-1">The agent builds memory as you chat</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
