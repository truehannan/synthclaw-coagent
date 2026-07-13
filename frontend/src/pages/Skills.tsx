import { useEffect, useState } from "react";
import { skills as api, mcp } from "@/lib/api";
import { Download, Trash2, RefreshCw, Plug, Plus } from "lucide-react";

interface McpServer {
  name: string;
  config: any;
  transport: string;
}

export default function Skills() {
  const [skillsList, setSkillsList] = useState<any[]>([]);
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);

  // MCP state
  const [mcpServers, setMcpServers] = useState<McpServer[]>([]);
  const [mcpJson, setMcpJson] = useState("");
  const [mcpError, setMcpError] = useState("");
  const [showMcpForm, setShowMcpForm] = useState(false);

  useEffect(() => { load(); loadMcp(); }, []);

  async function load() {
    try {
      const res = await api.list();
      setSkillsList(res.skills || []);
    } catch { setSkillsList([]); }
  }

  async function loadMcp() {
    try {
      const res = await mcp.list();
      setMcpServers(res.servers || []);
    } catch { setMcpServers([]); }
  }

  async function install() {
    if (!source) return;
    setLoading(true);
    try {
      await api.install(source);
      setSource("");
      load();
    } catch {}
    setLoading(false);
  }

  async function uninstall(name: string) {
    await api.uninstall(name);
    load();
  }

  async function reinstallAll() {
    setLoading(true);
    await api.reinstall();
    load();
    setLoading(false);
  }

  async function addMcpServer() {
    setMcpError("");
    if (!mcpJson.trim()) { setMcpError("Paste JSON config"); return; }
    try {
      const parsed = JSON.parse(mcpJson.trim());
      await mcp.add(parsed);
      setMcpJson("");
      setShowMcpForm(false);
      loadMcp();
    } catch (err: any) {
      setMcpError(err.message || "Invalid JSON");
    }
  }

  async function removeMcpServer(name: string) {
    await mcp.remove(name);
    loadMcp();
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Skills</h1>
      <p className="mb-6 text-xs text-muted">Install domain skills from ClawHub or URLs</p>

      {/* Install form */}
      <div className="mb-6 flex gap-2">
        <input value={source} onChange={(e) => setSource(e.target.value)}
          placeholder="@user/skill or https://..." 
          className="flex-1 rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
        <button onClick={install} disabled={!source || loading}
          className="flex items-center gap-1 rounded-sm bg-primary px-3 py-2 text-xs text-white hover:bg-primary-hover disabled:opacity-30">
          <Download className="h-3.5 w-3.5" /> Install
        </button>
        <button onClick={reinstallAll} disabled={loading}
          className="flex items-center gap-1 rounded-sm border border-border px-3 py-2 text-xs text-muted hover:border-primary hover:text-primary">
          <RefreshCw className="h-3.5 w-3.5" /> Reinstall All
        </button>
      </div>

      {/* Skills list */}
      <div className="space-y-1 mb-10">
        {skillsList.map((s: any, i) => (
          <div key={i} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5">
            <div>
              <span className="text-xs font-semibold text-foreground">{s.name || s}</span>
              {s.source && <span className="ml-2 text-[10px] text-muted">from {s.source}</span>}
            </div>
            <button onClick={() => uninstall(s.name || s)} className="text-muted hover:text-danger">
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
        {skillsList.length === 0 && (
          <p className="text-center text-xs text-muted py-8">No skills installed. Try @devops/nginx-expert</p>
        )}
      </div>

      {/* MCP Servers section */}
      <div className="border-t border-border pt-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="flex items-center gap-2">
              <Plug className="h-4 w-4 text-primary" />
              <h2 className="text-sm font-bold">MCP Servers</h2>
            </div>
            <p className="mt-1 text-[10px] text-muted">Model Context Protocol servers for tool integrations</p>
          </div>
          <button
            onClick={() => setShowMcpForm(!showMcpForm)}
            className="flex items-center gap-1 rounded-sm bg-primary px-3 py-1.5 text-xs text-white hover:bg-primary-hover"
          >
            <Plus className="h-3.5 w-3.5" /> Add Server
          </button>
        </div>

        {/* Add MCP form */}
        {showMcpForm && (
          <div className="mb-4 rounded-sm border border-border bg-card p-4 space-y-3">
            <p className="text-[10px] text-muted">
              Paste MCP JSON config. Supports formats:
              <code className="ml-1 text-primary">{"{ \"mcpServers\": { ... } }"}</code> or direct server configs.
            </p>
            <textarea
              value={mcpJson}
              onChange={(e) => setMcpJson(e.target.value)}
              placeholder={`{\n  "mcpServers": {\n    "server-name": {\n      "command": "npx",\n      "args": ["-y", "@modelcontextprotocol/server-filesystem"]\n    }\n  }\n}`}
              rows={6}
              className="w-full resize-none rounded-sm border border-border bg-background px-3 py-2 text-xs text-foreground font-mono placeholder-muted/40 outline-none focus:border-primary"
            />
            {mcpError && <p className="text-[10px] text-danger">{mcpError}</p>}
            <div className="flex gap-2">
              <button onClick={addMcpServer}
                className="rounded-sm bg-primary px-4 py-1.5 text-xs font-semibold text-white hover:bg-primary-hover">
                Save
              </button>
              <button onClick={() => { setShowMcpForm(false); setMcpError(""); setMcpJson(""); }}
                className="rounded-sm border border-border px-4 py-1.5 text-xs text-muted hover:text-foreground">
                Cancel
              </button>
            </div>
          </div>
        )}

        {/* MCP servers list */}
        <div className="space-y-1">
          {mcpServers.map((s) => (
            <div key={s.name} className="flex items-center justify-between rounded-sm border border-border bg-card px-4 py-2.5">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-foreground">{s.name}</span>
                  <span className="rounded-full bg-primary/10 px-1.5 py-0.5 text-[9px] text-primary font-medium">
                    {s.transport}
                  </span>
                </div>
                {s.config.command && (
                  <p className="mt-0.5 text-[10px] text-muted truncate">
                    {s.config.command} {(s.config.args || []).join(" ")}
                  </p>
                )}
                {s.config.url && (
                  <p className="mt-0.5 text-[10px] text-muted truncate">{s.config.url}</p>
                )}
              </div>
              <button onClick={() => removeMcpServer(s.name)} className="text-muted hover:text-danger ml-2">
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </div>
          ))}
          {mcpServers.length === 0 && !showMcpForm && (
            <p className="text-center text-xs text-muted py-6">
              No MCP servers configured. Click "Add Server" to connect one.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
