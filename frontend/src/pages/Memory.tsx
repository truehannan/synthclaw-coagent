import { useEffect, useState } from "react";
import { memory as api } from "@/lib/api";
import { Plus, Trash2 } from "lucide-react";

export default function Memory() {
  const [facts, setFacts] = useState<Record<string, string>>({});
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  useEffect(() => { load(); }, []);

  async function load() {
    const res = await api.all();
    setFacts(res.facts || {});
  }

  async function add() {
    if (!newKey || !newValue) return;
    await api.set(newKey, newValue);
    setNewKey(""); setNewValue("");
    load();
  }

  async function remove(key: string) {
    await api.del(key);
    load();
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Memory</h1>
      <p className="mb-6 text-xs text-muted">Facts the agent remembers</p>

      {/* Add form */}
      <div className="mb-6 flex gap-2">
        <input value={newKey} onChange={(e) => setNewKey(e.target.value)} placeholder="key"
          className="w-32 rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
        <input value={newValue} onChange={(e) => setNewValue(e.target.value)} placeholder="value"
          className="flex-1 rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
        <button onClick={add} disabled={!newKey || !newValue}
          className="rounded-sm bg-primary px-3 py-2 text-xs text-white hover:bg-primary-hover disabled:opacity-30">
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Facts list */}
      <div className="space-y-1">
        {Object.entries(facts).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5">
            <div className="flex-1 min-w-0">
              <span className="text-xs font-semibold text-primary">{key}</span>
              <span className="mx-2 text-xs text-muted">=</span>
              <span className="text-xs text-foreground truncate">{value}</span>
            </div>
            <button onClick={() => remove(key)} className="ml-2 text-muted hover:text-danger">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
        {Object.keys(facts).length === 0 && (
          <p className="text-center text-xs text-muted py-8">No memory facts stored yet</p>
        )}
      </div>
    </div>
  );
}
