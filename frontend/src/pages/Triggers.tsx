import { useEffect, useState } from "react";
import { composio } from "@/lib/api";
import { Zap, Trash2, Plus, Loader2, RefreshCw, Search } from "lucide-react";

interface Trigger {
  id: string;
  slug: string;
  status: string;
  app: string;
  config: any;
  created: string;
}

export default function Triggers() {
  const [triggers, setTriggers] = useState<Trigger[]>([]);
  const [loading, setLoading] = useState(true);
  const [available, setAvailable] = useState(false);
  const [deleting, setDeleting] = useState("");

  // Create form
  const [showCreate, setShowCreate] = useState(false);
  const [newSlug, setNewSlug] = useState("");
  const [newConfig, setNewConfig] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState("");

  useEffect(() => { load(); }, []);

  async function load() {
    setLoading(true);
    try {
      const res = await composio.triggers.list();
      setTriggers(res.triggers || []);
      setAvailable(res.available !== false);
    } catch {
      setAvailable(false);
    }
    setLoading(false);
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this trigger? It will stop listening for events.")) return;
    setDeleting(id);
    try {
      await composio.triggers.del(id);
      load();
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
    setDeleting("");
  }

  async function handleCreate() {
    if (!newSlug.trim()) { setCreateError("Trigger slug is required"); return; }
    setCreating(true);
    setCreateError("");
    try {
      let config = undefined;
      if (newConfig.trim()) {
        try { config = JSON.parse(newConfig); }
        catch { setCreateError("Invalid JSON in config"); setCreating(false); return; }
      }
      const res = await composio.triggers.create(newSlug.trim().toUpperCase(), config);
      if (res.success) {
        setNewSlug("");
        setNewConfig("");
        setShowCreate(false);
        load();
      } else {
        setCreateError(res.error || "Failed to create trigger");
      }
    } catch (err: any) {
      setCreateError(err.message || "Failed");
    }
    setCreating(false);
  }

  if (!available && !loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center max-w-sm">
          <Zap className="h-10 w-10 text-muted mx-auto mb-3" />
          <p className="text-sm text-muted">Composio not configured</p>
          <p className="text-[10px] text-muted mt-2">
            Add your COMPOSIO_API_KEY in the Setup Wizard to enable triggers and automations.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-border bg-card px-4 py-3">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-sm font-bold">[+] Triggers</h1>
            <p className="text-[9px] text-muted">
              Automations that listen for events (new email, new commit, etc.) and fire actions
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={load} className="rounded-sm border border-border p-1.5 text-muted hover:text-foreground hover:border-primary">
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
            <button onClick={() => setShowCreate(!showCreate)}
              className="flex items-center gap-1.5 rounded-sm bg-primary px-3 py-1.5 text-xs text-white hover:bg-primary-hover">
              <Plus className="h-3.5 w-3.5" /> New Trigger
            </button>
          </div>
        </div>
      </div>

      {/* Create form */}
      {showCreate && (
        <div className="border-b border-border bg-card/50 px-4 py-4">
          <div className="max-w-lg space-y-3">
            <p className="text-[10px] text-muted">
              Create a trigger by providing its slug (e.g. GMAIL_NEW_EMAIL, GITHUB_PUSH_EVENT, SLACK_NEW_MESSAGE).
              Use the agent to discover available triggers: ask "what triggers are available for gmail?"
            </p>
            <div>
              <label className="text-[9px] text-muted block mb-1">Trigger Slug *</label>
              <input value={newSlug} onChange={e => setNewSlug(e.target.value)}
                placeholder="GMAIL_NEW_EMAIL"
                className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary font-mono uppercase" />
            </div>
            <div>
              <label className="text-[9px] text-muted block mb-1">Config (optional JSON)</label>
              <textarea value={newConfig} onChange={e => setNewConfig(e.target.value)}
                placeholder='{"filter": "from:important@example.com"}'
                rows={3}
                className="w-full resize-none rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary font-mono" />
            </div>
            {createError && <p className="text-[10px] text-danger">{createError}</p>}
            <div className="flex gap-2">
              <button onClick={() => { setShowCreate(false); setCreateError(""); }}
                className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Cancel</button>
              <button onClick={handleCreate} disabled={creating || !newSlug.trim()}
                className="rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                {creating ? "Creating..." : "Create Trigger"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Triggers list */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-5 w-5 animate-spin text-muted" />
          </div>
        ) : triggers.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center py-12">
              <Zap className="h-8 w-8 text-muted/30 mx-auto mb-3" />
              <p className="text-xs text-muted font-medium">No triggers active</p>
              <p className="text-[9px] text-muted/60 mt-1 max-w-[250px] mx-auto">
                Create a trigger to start listening for events from your connected apps.
                Or ask the agent: "create a trigger for new emails"
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-2 max-w-2xl">
            {triggers.map(t => (
              <div key={t.id} className="rounded-sm border border-border bg-card p-4 transition-colors hover:border-muted">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <img src={`https://logos.composio.dev/api/${t.app || t.slug.split("_")[0].toLowerCase()}`}
                        alt="" className="h-4 w-4 rounded-sm"
                        onError={e => (e.currentTarget.style.display = "none")} />
                      <span className="text-xs font-bold text-foreground font-mono">{t.slug}</span>
                      <span className={`rounded-full px-1.5 py-0.5 text-[8px] font-medium ${
                        t.status === "active" ? "bg-success/15 text-success" :
                        t.status === "disabled" ? "bg-muted/15 text-muted" :
                        "bg-amber-400/15 text-amber-400"
                      }`}>{t.status}</span>
                    </div>
                    <div className="flex items-center gap-3 mt-1.5 text-[9px] text-muted">
                      {t.app && <span>App: {t.app}</span>}
                      {t.created && <span>Created: {t.created}</span>}
                      <span className="font-mono text-muted/50 truncate">{t.id}</span>
                    </div>
                    {t.config && Object.keys(t.config).length > 0 && (
                      <pre className="mt-2 rounded-sm bg-background border border-border/50 px-2 py-1 text-[8px] text-muted overflow-x-auto max-w-full">
                        {JSON.stringify(t.config, null, 2)}
                      </pre>
                    )}
                  </div>
                  <button onClick={() => handleDelete(t.id)}
                    disabled={deleting === t.id}
                    className="rounded-sm border border-border p-1.5 text-muted hover:text-danger hover:border-danger flex-shrink-0">
                    {deleting === t.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
