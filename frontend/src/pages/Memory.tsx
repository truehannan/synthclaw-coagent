import { useEffect, useState } from "react";
import { memory as api } from "@/lib/api";
import { Plus, Trash2, Edit3, Check, X, GitBranch, List } from "lucide-react";

type ViewMode = "tree" | "list";

interface MemoryNode {
  key: string;
  value: string;
  category: string;
  children: MemoryNode[];
}

function categorize(facts: Record<string, string>): MemoryNode[] {
  // Group facts by prefix (before first _ or :)
  const groups: Record<string, { key: string; value: string }[]> = {};
  for (const [key, value] of Object.entries(facts)) {
    const sep = key.indexOf("_") > 0 ? key.indexOf("_") : key.indexOf(":") > 0 ? key.indexOf(":") : -1;
    const category = sep > 0 ? key.slice(0, sep) : "general";
    if (!groups[category]) groups[category] = [];
    groups[category].push({ key, value });
  }
  return Object.entries(groups).map(([category, items]) => ({
    key: category,
    value: `${items.length} items`,
    category,
    children: items.map(i => ({ key: i.key, value: i.value, category, children: [] })),
  }));
}

export default function Memory() {
  const [facts, setFacts] = useState<Record<string, string>>({});
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [editKey, setEditKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [view, setView] = useState<ViewMode>("tree");

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      const res = await api.all();
      setFacts(res.facts || {});
    } catch {}
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
    load();
  }

  async function remove(key: string) {
    await api.del(key);
    load();
  }

  const tree = categorize(facts);
  const totalFacts = Object.keys(facts).length;

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-lg font-bold">[+] Memory</h1>
          <p className="text-xs text-muted">{totalFacts} facts stored — agent's knowledge base</p>
        </div>
        <div className="flex items-center gap-1 rounded-sm border border-border p-0.5">
          <button onClick={() => setView("tree")}
            className={`rounded-sm p-1.5 ${view === "tree" ? "bg-primary/10 text-primary" : "text-muted hover:text-foreground"}`}>
            <GitBranch className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => setView("list")}
            className={`rounded-sm p-1.5 ${view === "list" ? "bg-primary/10 text-primary" : "text-muted hover:text-foreground"}`}>
            <List className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Add form */}
      <div className="mb-6 flex gap-2">
        <input value={newKey} onChange={(e) => setNewKey(e.target.value)} placeholder="key"
          className="w-40 rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
        <input value={newValue} onChange={(e) => setNewValue(e.target.value)} placeholder="value"
          onKeyDown={e => e.key === "Enter" && add()}
          className="flex-1 rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
        <button onClick={add} disabled={!newKey || !newValue}
          className="rounded-sm bg-primary px-3 py-2 text-xs text-white hover:bg-primary-hover disabled:opacity-30">
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Tree View */}
      {view === "tree" && (
        <div className="space-y-3">
          {tree.map(branch => (
            <div key={branch.key} className="rounded-sm border border-border bg-card overflow-hidden">
              <div className="flex items-center gap-2 bg-card-hover px-4 py-2 border-b border-border">
                <GitBranch className="h-3 w-3 text-primary" />
                <span className="text-xs font-semibold text-primary">{branch.key}</span>
                <span className="text-[9px] text-muted ml-auto">{branch.children.length}</span>
              </div>
              <div className="divide-y divide-border/50">
                {branch.children.map(node => (
                  <div key={node.key} className="flex items-center gap-2 px-4 py-2 group">
                    <div className="flex items-center gap-1 text-muted">
                      <span className="text-border">├─</span>
                    </div>
                    {editKey === node.key ? (
                      <div className="flex-1 flex gap-2">
                        <input value={editValue} onChange={e => setEditValue(e.target.value)} autoFocus
                          onKeyDown={e => { if (e.key === "Enter") update(node.key); if (e.key === "Escape") setEditKey(null); }}
                          className="flex-1 rounded-sm border border-primary bg-background px-2 py-1 text-xs outline-none" />
                        <button onClick={() => update(node.key)} className="text-success"><Check className="h-3.5 w-3.5" /></button>
                        <button onClick={() => setEditKey(null)} className="text-muted"><X className="h-3.5 w-3.5" /></button>
                      </div>
                    ) : (
                      <>
                        <span className="text-[10px] font-medium text-foreground">{node.key.includes("_") ? node.key.split("_").slice(1).join("_") : node.key}</span>
                        <span className="text-[10px] text-muted mx-1">=</span>
                        <span className="text-[10px] text-foreground flex-1 truncate">{node.value}</span>
                        <button onClick={() => { setEditKey(node.key); setEditValue(node.value); }}
                          className="opacity-0 group-hover:opacity-100 text-muted hover:text-primary"><Edit3 className="h-3 w-3" /></button>
                        <button onClick={() => remove(node.key)}
                          className="opacity-0 group-hover:opacity-100 text-muted hover:text-danger"><Trash2 className="h-3 w-3" /></button>
                      </>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
          {tree.length === 0 && (
            <div className="text-center py-12">
              <GitBranch className="h-8 w-8 text-muted mx-auto mb-3" />
              <p className="text-xs text-muted">No memories yet</p>
              <p className="text-[10px] text-muted mt-1">The agent builds memory as you chat</p>
            </div>
          )}
        </div>
      )}

      {/* List View */}
      {view === "list" && (
        <div className="space-y-1">
          {Object.entries(facts).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5 group">
              {editKey === key ? (
                <div className="flex-1 flex gap-2">
                  <input value={editValue} onChange={e => setEditValue(e.target.value)} autoFocus
                    onKeyDown={e => { if (e.key === "Enter") update(key); if (e.key === "Escape") setEditKey(null); }}
                    className="flex-1 rounded-sm border border-primary bg-background px-2 py-1 text-xs outline-none" />
                  <button onClick={() => update(key)} className="text-success"><Check className="h-3.5 w-3.5" /></button>
                  <button onClick={() => setEditKey(null)} className="text-muted"><X className="h-3.5 w-3.5" /></button>
                </div>
              ) : (
                <>
                  <div className="flex-1 min-w-0">
                    <span className="text-xs font-semibold text-primary">{key}</span>
                    <span className="mx-2 text-xs text-muted">=</span>
                    <span className="text-xs text-foreground truncate">{value}</span>
                  </div>
                  <div className="flex gap-1 ml-2">
                    <button onClick={() => { setEditKey(key); setEditValue(value); }}
                      className="opacity-0 group-hover:opacity-100 text-muted hover:text-primary"><Edit3 className="h-3.5 w-3.5" /></button>
                    <button onClick={() => remove(key)}
                      className="opacity-0 group-hover:opacity-100 text-muted hover:text-danger"><Trash2 className="h-3.5 w-3.5" /></button>
                  </div>
                </>
              )}
            </div>
          ))}
          {Object.keys(facts).length === 0 && (
            <p className="text-center text-xs text-muted py-8">No memory facts stored yet</p>
          )}
        </div>
      )}
    </div>
  );
}
