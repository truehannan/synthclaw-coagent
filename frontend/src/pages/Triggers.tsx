import { useEffect, useState } from "react";
import { composio } from "@/lib/api";
import { Zap, Square, Loader2, RefreshCw } from "lucide-react";

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
  const [stopping, setStopping] = useState("");

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

  async function handleStop(id: string) {
    if (!confirm("Stop this trigger? It will no longer listen for events.")) return;
    setStopping(id);
    try {
      await composio.triggers.del(id);
      load();
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
    setStopping("");
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
              Active automations listening for events. Ask the agent to create or edit triggers.
            </p>
          </div>
          <button onClick={load} className="rounded-sm border border-border p-1.5 text-muted hover:text-foreground hover:border-primary">
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

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
              <p className="text-[9px] text-muted/60 mt-1 max-w-[280px] mx-auto">
                Ask the agent to create triggers for you. Try: "Create a trigger that notifies me on new emails" or "Set up automation for GitHub push events"
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
                  <button onClick={() => handleStop(t.id)}
                    disabled={stopping === t.id}
                    title="Stop trigger"
                    className="flex items-center gap-1 rounded-sm border border-border px-2 py-1.5 text-[9px] text-muted hover:text-danger hover:border-danger flex-shrink-0">
                    {stopping === t.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Square className="h-3 w-3" />}
                    <span>Stop</span>
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
