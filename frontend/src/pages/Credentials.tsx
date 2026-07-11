import { useEffect, useState } from "react";
import { credentials as api } from "@/lib/api";
import { Plus, Trash2, Eye, EyeOff } from "lucide-react";

export default function Credentials() {
  const [creds, setCreds] = useState<Array<{ name: string; description?: string }>>([]);
  const [newName, setNewName] = useState("");
  const [newValue, setNewValue] = useState("");
  const [newDesc, setNewDesc] = useState("");

  useEffect(() => { load(); }, []);

  async function load() {
    const res = await api.list();
    setCreds(res.credentials || []);
  }

  async function add() {
    if (!newName || !newValue) return;
    await api.store(newName, newValue, newDesc);
    setNewName(""); setNewValue(""); setNewDesc("");
    load();
  }

  async function remove(name: string) {
    if (!confirm(`Delete credential "${name}"?`)) return;
    await api.del(name);
    load();
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Credentials</h1>
      <p className="mb-6 text-xs text-muted">Encrypted credential store (values never shown)</p>

      {/* Add form */}
      <div className="mb-6 space-y-2">
        <div className="flex gap-2">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Name (e.g. GITHUB_PAT)"
            className="w-48 rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
          <input type="password" value={newValue} onChange={(e) => setNewValue(e.target.value)} placeholder="Value"
            className="flex-1 rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
          <button onClick={add} disabled={!newName || !newValue}
            className="rounded-sm bg-primary px-3 py-2 text-xs text-white hover:bg-primary-hover disabled:opacity-30">
            <Plus className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Credentials list */}
      <div className="space-y-1">
        {creds.map((c) => (
          <div key={c.name} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5">
            <div className="flex items-center gap-3">
              <EyeOff className="h-3.5 w-3.5 text-muted" />
              <span className="text-xs font-semibold text-foreground">{c.name}</span>
              <span className="text-[10px] text-muted">••••••••</span>
            </div>
            <button onClick={() => remove(c.name)} className="text-muted hover:text-danger">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
        {creds.length === 0 && (
          <p className="text-center text-xs text-muted py-8">No credentials stored</p>
        )}
      </div>
    </div>
  );
}
