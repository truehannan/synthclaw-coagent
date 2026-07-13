import { useEffect, useState } from "react";
import { apis, composio } from "@/lib/api";
import { Link2, Unplug, Globe, Zap } from "lucide-react";
import Mascot from "@/components/Mascot";

interface ApiEntry {
  name: string;
  base_url: string;
  description: string;
  enabled: boolean;
  auth_cred: string;
}

interface ComposioConnection {
  app?: string;
  name?: string;
  status?: string;
  id?: string;
}

export default function Integrations() {
  const [apiList, setApiList] = useState<ApiEntry[]>([]);
  const [connections, setConnections] = useState<ComposioConnection[]>([]);
  const [composioAvailable, setComposioAvailable] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([loadApis(), loadComposio()]).finally(() => setLoading(false));
  }, []);

  async function loadApis() {
    try {
      const res = await apis.list();
      setApiList(res.apis || []);
    } catch { setApiList([]); }
  }

  async function loadComposio() {
    try {
      const res = await composio.connections();
      setConnections(res.connections || []);
      setComposioAvailable(res.available !== false);
    } catch {
      setComposioAvailable(false);
    }
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Integrations</h1>
      <p className="mb-6 text-xs text-muted">Connected apps and registered APIs</p>

      {/* Composio Section */}
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <Zap className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-bold">Composio</h2>
          {!composioAvailable && (
            <span className="rounded-full bg-muted/20 px-2 py-0.5 text-[9px] text-muted">Not configured</span>
          )}
        </div>

        {composioAvailable ? (
          connections.length > 0 ? (
            <div className="space-y-1">
              {connections.map((c, i) => (
                <div key={i} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5">
                  <div className="flex items-center gap-3">
                    <Link2 className="h-3.5 w-3.5 text-success" />
                    <span className="text-xs font-semibold text-foreground">{c.app || c.name || "Unknown"}</span>
                    {c.status && (
                      <span className={`rounded-full px-1.5 py-0.5 text-[9px] font-medium ${
                        c.status === "active" ? "bg-success/20 text-success" : "bg-muted/20 text-muted"
                      }`}>
                        {c.status}
                      </span>
                    )}
                  </div>
                  {c.id && <span className="text-[10px] text-muted">{c.id.slice(0, 8)}...</span>}
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-sm border border-border bg-card p-6 text-center">
              <Unplug className="mx-auto h-6 w-6 text-muted mb-2" />
              <p className="text-xs text-muted">No Composio connections yet.</p>
              <p className="text-[10px] text-muted mt-1">
                Use <code className="text-primary">/composio connect &lt;app&gt;</code> in chat to connect an app.
              </p>
            </div>
          )
        ) : (
          <div className="rounded-sm border border-border bg-card p-6 text-center">
            <p className="text-xs text-muted">Composio not configured.</p>
            <p className="text-[10px] text-muted mt-1">
              Set <code className="text-primary">COMPOSIO_API_KEY</code> in your environment to enable 200+ app integrations.
            </p>
          </div>
        )}
      </div>

      {/* Registered APIs Section */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Globe className="h-4 w-4 text-primary" />
          <h2 className="text-sm font-bold">Registered APIs</h2>
        </div>

        {apiList.length > 0 ? (
          <div className="space-y-1">
            {apiList.map((a) => (
              <div key={a.name} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-foreground">{a.name}</span>
                    <span className={`rounded-full px-1.5 py-0.5 text-[9px] font-medium ${
                      a.enabled ? "bg-success/20 text-success" : "bg-muted/20 text-muted"
                    }`}>
                      {a.enabled ? "active" : "disabled"}
                    </span>
                  </div>
                  <p className="mt-0.5 text-[10px] text-muted truncate">
                    {a.base_url}
                    {a.description && ` — ${a.description}`}
                  </p>
                </div>
                <span className="text-[10px] text-muted ml-2">cred: {a.auth_cred}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-sm border border-border bg-card p-6 text-center">
            <Mascot className="mx-auto mb-3 opacity-50" />
            <p className="text-xs text-muted">No APIs registered.</p>
            <p className="text-[10px] text-muted mt-1">
              Use <code className="text-primary">/api register &lt;name&gt; &lt;base_url&gt;</code> in chat to add one.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
