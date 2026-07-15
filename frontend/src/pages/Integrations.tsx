import { useEffect, useState, useRef, useCallback } from "react";
import { apis, composio } from "@/lib/api";
import { Link2, Unplug, Globe, Zap, Search, ExternalLink, Loader2 } from "lucide-react";

interface ComposioTool {
  slug: string;
  name: string;
  description: string;
  logo?: string;
  toolkit?: { slug: string; name: string; logo: string };
  tags?: string[];
  no_auth?: boolean;
}

interface ApiEntry {
  name: string;
  base_url: string;
  description: string;
  enabled: boolean;
  auth_cred: string;
}

export default function Integrations() {
  const [apiList, setApiList] = useState<ApiEntry[]>([]);
  const [composioAvailable, setComposioAvailable] = useState(false);
  const [tools, setTools] = useState<ComposioTool[]>([]);
  const [toolsSearch, setToolsSearch] = useState("");
  const [toolsPage, setToolsPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [totalItems, setTotalItems] = useState(0);
  const [loadingTools, setLoadingTools] = useState(false);
  const [connecting, setConnecting] = useState("");
  const [connectedSlugs, setConnectedSlugs] = useState<Set<string>>(new Set());
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const loadedRef = useRef(false);

  useEffect(() => {
    if (!loadedRef.current) {
      loadedRef.current = true;
      loadApis();
      loadTools(1, "");
      loadConnections();
    }
  }, []);

  async function loadConnections() {
    try {
      const res = await composio.connections();
      const slugs = new Set<string>();
      for (const c of (res.connections || [])) {
        if (c.slug) slugs.add(c.slug.toLowerCase());
        if (c.app) slugs.add(c.app.toLowerCase());
      }
      setConnectedSlugs(slugs);
      setComposioAvailable(res.available !== false);
    } catch {}
  }

  async function loadApis() {
    try { const res = await apis.list(); setApiList(res.apis || []); } catch {}
  }

  async function loadTools(page: number, search: string) {
    setLoadingTools(true);
    try {
      const res = await composio.tools(page, search);
      setComposioAvailable(res.available !== false);
      setTools(res.items || []);
      setTotalPages(res.total_pages || 0);
      setTotalItems(res.total_items || 0);
      setToolsPage(page);
    } catch {
      setComposioAvailable(false);
    } finally {
      setLoadingTools(false);
    }
  }

  function handleSearch(value: string) {
    setToolsSearch(value);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(() => loadTools(1, value), 400);
  }

  async function handleConnect(toolkit: string) {
    setConnecting(toolkit);
    try {
      const res = await composio.connect(toolkit);
      if (res.redirectUrl) {
        window.open(res.redirectUrl, "_blank");
        // Poll for connection completion after redirect
        setTimeout(() => loadConnections(), 5000);
        setTimeout(() => loadConnections(), 15000);
      } else if (res.error) {
        alert(`Connection failed: ${res.error}\n${res.detail || ""}`);
      } else {
        // Might have connected directly (API key auth)
        loadConnections();
      }
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
    setConnecting("");
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-border bg-card px-4 py-3">
        <h1 className="text-sm font-bold">[+] Integrations</h1>
        <p className="text-[9px] text-muted">
          {composioAvailable
            ? `${totalItems} tools available — connect apps for your agent`
            : "Configure COMPOSIO_API_KEY in setup to enable 1000+ integrations"}
        </p>
      </div>

      {composioAvailable ? (
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Search */}
          <div className="px-4 py-3 border-b border-border">
            <div className="relative max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted" />
              <input value={toolsSearch} onChange={e => handleSearch(e.target.value)}
                placeholder="Search tools (gmail, github, slack, notion...)"
                className="w-full rounded-sm border border-border bg-background pl-9 pr-3 py-2 text-xs outline-none focus:border-primary" />
            </div>
          </div>

          {/* Tools grid */}
          <div className="flex-1 overflow-y-auto p-4">
            {loadingTools && tools.length === 0 && (
              <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div key={i} className="rounded-sm border border-border bg-card p-3 animate-pulse">
                    <div className="flex items-start gap-2">
                      <div className="h-5 w-5 rounded-sm bg-muted/20" />
                      <div className="flex-1 space-y-1.5">
                        <div className="h-3 w-24 rounded bg-muted/20" />
                        <div className="h-2 w-full rounded bg-muted/10" />
                        <div className="h-2 w-3/4 rounded bg-muted/10" />
                      </div>
                    </div>
                    <div className="mt-2 flex gap-1">
                      <div className="h-3 w-10 rounded-full bg-muted/10" />
                      <div className="h-3 w-8 rounded-full bg-muted/10" />
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">
              {tools.map(tool => {
                const logo = tool.logo || tool.toolkit?.logo || "";
                const name = tool.name || tool.toolkit?.name || tool.slug || "";
                const slug = tool.slug || "";
                const desc = tool.description || "";
                const tags: string[] = tool.tags || [];
                const isConnected = connectedSlugs.has(slug.toLowerCase()) || connectedSlugs.has(name.toLowerCase());
                return (
                <div key={slug} className={`rounded-sm border bg-card p-3 transition-colors ${isConnected ? "border-success/30" : "border-border hover:border-muted"}`}>
                  <div className="flex items-start gap-2">
                    {logo && (
                      <img src={logo} alt="" className="h-5 w-5 rounded-sm flex-shrink-0 mt-0.5" onError={e => (e.currentTarget.style.display = "none")} />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <p className="text-[10px] font-semibold text-foreground truncate">{name}</p>
                        {isConnected && <span className="rounded-full bg-success/15 px-1.5 py-0.5 text-[8px] text-success font-medium">Connected</span>}
                      </div>
                      <p className="text-[9px] text-muted mt-0.5 line-clamp-2">{desc}</p>
                    </div>
                  </div>
                  {tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-2">
                      {tags.slice(0, 3).map(tag => (
                        <span key={tag} className="rounded-full bg-muted/10 px-1.5 py-0.5 text-[8px] text-muted">{tag}</span>
                      ))}
                    </div>
                  )}
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-[8px] text-muted font-mono truncate max-w-[120px]">{slug}</span>
                    {isConnected ? (
                      <span className="text-[9px] text-success font-medium">Active</span>
                    ) : (
                      <button onClick={() => handleConnect(slug)}
                        disabled={connecting === slug}
                        className="flex items-center gap-1 rounded-sm bg-primary/10 px-2 py-0.5 text-[9px] text-primary hover:bg-primary/20 disabled:opacity-50">
                        {connecting === slug
                          ? <Loader2 className="h-2.5 w-2.5 animate-spin" />
                          : <ExternalLink className="h-2.5 w-2.5" />}
                        Connect
                      </button>
                    )}
                  </div>
                </div>
                );
              })}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-4 py-3">
                <button onClick={() => loadTools(toolsPage - 1, toolsSearch)} disabled={toolsPage <= 1}
                  className="rounded-sm border border-border px-3 py-1 text-[10px] text-muted hover:text-foreground disabled:opacity-30">Prev</button>
                <span className="text-[10px] text-muted">Page {toolsPage} of {totalPages}</span>
                <button onClick={() => loadTools(toolsPage + 1, toolsSearch)} disabled={toolsPage >= totalPages}
                  className="rounded-sm border border-border px-3 py-1 text-[10px] text-muted hover:text-foreground disabled:opacity-30">Next</button>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center max-w-sm">
            <Zap className="h-10 w-10 text-muted mx-auto mb-3" />
            <p className="text-sm text-muted">Composio not configured</p>
            <p className="text-[10px] text-muted mt-2">
              Add your COMPOSIO_API_KEY in the Setup Wizard or Settings to enable 1000+ app integrations (Gmail, GitHub, Slack, Notion, etc.)
            </p>
          </div>
        </div>
      )}

      {/* Registered APIs section */}
      {apiList.length > 0 && (
        <div className="border-t border-border p-4">
          <div className="flex items-center gap-2 mb-2">
            <Globe className="h-3.5 w-3.5 text-primary" />
            <span className="text-[10px] font-semibold text-muted">Registered APIs ({apiList.length})</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {apiList.map(a => (
              <div key={a.name} className="rounded-sm border border-border bg-card px-3 py-1.5 text-[9px]">
                <span className="font-medium text-foreground">{a.name}</span>
                <span className="text-muted ml-1">({a.base_url})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
