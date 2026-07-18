import { useState } from "react";
import { ChevronDown, ChevronRight, Terminal, Brain, ListChecks, Loader2, AlertCircle, CheckCircle2, ExternalLink } from "lucide-react";

// ── Event Types ──────────────────────────────────────────────────────────────

export interface ChatEvent {
  type: "thinking" | "tool_call" | "tool_result" | "agent_step" | "agent_spawn" | "plan" | "text" | "progress" | "done" | "error" | "connect_required";
  content?: string;
  full?: string;
  message?: string;
  tool?: string;
  args?: any;
  output?: string;
  app?: string;
  app_name?: string;
  agent?: { id: string; role: string; task: string };
  steps?: { agent: string; task: string }[];
}

// ── Agent Colors ─────────────────────────────────────────────────────────────

const AGENT_COLORS: Record<string, string> = {
  orchestrator: "#ef4444",
  researcher: "#3b82f6",
  executor: "#10b981",
  reviewer: "#f59e0b",
  observer: "#8b5cf6",
  specialist: "#ec4899",
  planner: "#06b6d4",
  coder: "#22c55e",
};

// ── Components ───────────────────────────────────────────────────────────────

export function ThinkingIndicator({ content }: { content?: string }) {
  const [collapsed, setCollapsed] = useState(true);
  const text = content || "Processing...";
  const isLong = text.length > 100;

  return (
    <div className="animate-fade-in py-1">
      <div className="flex items-center gap-2 text-xs text-muted cursor-pointer" onClick={() => isLong && setCollapsed(!collapsed)}>
        <Loader2 className="h-3.5 w-3.5 animate-spin text-primary flex-shrink-0" />
        <span className={`${isLong && collapsed ? "line-clamp-2" : ""}`}>{isLong && collapsed ? text.slice(0, 100) + "..." : text}</span>
        {isLong && (
          <span className="text-[8px] text-muted/60 flex-shrink-0">{collapsed ? "▸" : "▾"}</span>
        )}
      </div>
    </div>
  );
}

// ── Tool label/icon mapping ──────────────────────────────────────────────────

const TOOL_ICONS: Record<string, string> = {
  "Web Search": "🔍",
  "Shell": "⚙️",
  "System": "🖥️",
  "Check Connection": "🔌",
  "Create Trigger": "⚡",
  "Delete Trigger": "🗑️",
  "List Triggers": "📋",
  "Discover Triggers": "🔍",
  "Discover Tools": "🔍",
};

function getToolDisplay(tool: string, app?: string): { label: string; icon: string | null; logoUrl: string | null } {
  // Known internal tools
  if (TOOL_ICONS[tool]) {
    return { label: tool, icon: TOOL_ICONS[tool], logoUrl: null };
  }
  // Composio tool slugs (GMAIL_SEND_EMAIL → "Gmail: Send Email")
  if (tool.includes("_") && tool === tool.toUpperCase()) {
    const parts = tool.split("_");
    const appName = parts[0].charAt(0) + parts[0].slice(1).toLowerCase();
    const action = parts.slice(1).map(p => p.charAt(0) + p.slice(1).toLowerCase()).join(" ");
    const slug = app || parts[0].toLowerCase();
    return { label: `${appName}: ${action}`, icon: null, logoUrl: `https://logos.composio.dev/api/${slug}` };
  }
  // Generic tool with app context
  if (app) {
    return { label: tool, icon: null, logoUrl: `https://logos.composio.dev/api/${app}` };
  }
  return { label: tool, icon: "🔧", logoUrl: null };
}

export function ToolCallCard({ tool, args, output, app, collapsed: initialCollapsed = true }: { tool: string; args?: any; output?: string; app?: string; collapsed?: boolean }) {
  const [collapsed, setCollapsed] = useState(initialCollapsed);
  const { label, icon, logoUrl } = getToolDisplay(tool, app);

  return (
    <div className="rounded-sm border border-border/60 bg-card/50 overflow-hidden animate-fade-in my-1">
      <button onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-card-hover">
        {collapsed ? <ChevronRight className="h-3 w-3 text-muted" /> : <ChevronDown className="h-3 w-3 text-muted" />}
        {logoUrl ? (
          <img src={logoUrl} alt="" className="h-3.5 w-3.5 rounded-sm" onError={e => { e.currentTarget.style.display = "none"; }} />
        ) : icon ? (
          <span className="text-[11px] leading-none">{icon}</span>
        ) : (
          <Terminal className="h-3 w-3 text-primary" />
        )}
        <span className="text-[10px] font-semibold text-primary truncate">{label}</span>
        {output && <CheckCircle2 className="h-3 w-3 text-success ml-auto flex-shrink-0" />}
        {!output && <Loader2 className="h-3 w-3 animate-spin text-muted ml-auto flex-shrink-0" />}
      </button>
      {!collapsed && (
        <div className="border-t border-border/40 px-3 py-2 space-y-1">
          {args && (
            <pre className="text-[9px] text-muted font-mono whitespace-pre-wrap break-all max-h-24 overflow-y-auto">
              {typeof args === "string" ? args : JSON.stringify(args, null, 2)}
            </pre>
          )}
          {output && (
            <pre className="text-[9px] text-foreground/80 font-mono whitespace-pre-wrap break-all max-h-32 overflow-y-auto border-t border-border/30 pt-1 mt-1">
              {output.length > 500 ? output.slice(0, 500) + "..." : output}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

export function AgentBadge({ role, task }: { role: string; task?: string }) {
  const color = AGENT_COLORS[role.toLowerCase()] || "#6b7280";
  return (
    <div className="flex items-center gap-2 py-1 animate-fade-in">
      <span className="h-2 w-2 rounded-full animate-pulse" style={{ background: color }} />
      <span className="text-[10px] font-semibold capitalize" style={{ color }}>{role}</span>
      {task && <span className="text-[9px] text-muted truncate max-w-[250px]">{task}</span>}
    </div>
  );
}

export function PlanCard({ steps }: { steps: { agent: string; task: string }[] }) {
  return (
    <div className="rounded-sm border border-border/60 bg-card/50 p-3 animate-fade-in my-1">
      <div className="flex items-center gap-2 mb-2">
        <ListChecks className="h-3.5 w-3.5 text-primary" />
        <span className="text-[10px] font-semibold text-primary">Plan ({steps.length} steps)</span>
      </div>
      <div className="space-y-1">
        {steps.map((step, i) => {
          const color = AGENT_COLORS[step.agent?.toLowerCase()] || "#6b7280";
          return (
            <div key={i} className="flex items-center gap-2 text-[9px]">
              <span className="text-muted w-4">{i + 1}.</span>
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
              <span className="capitalize font-medium" style={{ color }}>{step.agent}</span>
              <span className="text-muted truncate">{step.task}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ProgressCard({ content }: { content: string }) {
  return (
    <div className="flex items-center gap-2 text-[10px] text-muted py-1 animate-fade-in">
      <Brain className="h-3 w-3 text-primary" />
      <span>{content}</span>
    </div>
  );
}

export function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-sm border border-danger/30 bg-danger/5 px-3 py-2 animate-fade-in my-1">
      <div className="flex items-center gap-2">
        <AlertCircle className="h-3.5 w-3.5 text-danger" />
        <span className="text-[10px] font-semibold text-danger">Error</span>
      </div>
      <p className="text-[10px] text-danger/80 mt-1">{message}</p>
    </div>
  );
}



export function IntegrationCard({ name, slug, connected, onConnect }: { name: string; slug: string; connected: boolean; onConnect?: () => void }) {
  const logoUrl = `https://logos.composio.dev/api/${slug}`;
  return (
    <div className={`rounded-sm border ${connected ? "border-success/30 bg-success/5" : "border-border bg-card/50"} p-3 animate-fade-in my-1`}>
      <div className="flex items-center gap-3">
        <img src={logoUrl} alt="" className="h-5 w-5 rounded-sm" onError={e => (e.currentTarget.style.display = "none")} />
        <span className="text-[11px] font-semibold text-foreground">{name || slug}</span>
        {connected ? (
          <span className="ml-auto flex items-center gap-1 text-[9px] text-success"><CheckCircle2 className="h-3 w-3" /> Connected</span>
        ) : (
          <button onClick={onConnect}
            className="ml-auto flex items-center gap-1 rounded-sm bg-primary/10 px-2.5 py-1 text-[9px] text-primary hover:bg-primary/20">
            <ExternalLink className="h-3 w-3" /> Connect
          </button>
        )}
      </div>
    </div>
  );
}


export function ConnectRequiredCard({ app, appName, onConnect }: { app: string; appName: string; onConnect?: (slug: string) => void }) {
  const [connecting, setConnecting] = useState(false);
  const [connected, setConnected] = useState(false);
  const logoUrl = `https://logos.composio.dev/api/${app}`;

  async function handleConnect() {
    if (!app) return;
    setConnecting(true);
    try {
      // Import composio API dynamically to avoid circular deps
      const { composio } = await import("@/lib/api");
      const res = await composio.connect(app);
      if (res.redirectUrl) {
        window.open(res.redirectUrl, "_blank");
        // Poll until connected
        const poll = setInterval(async () => {
          try {
            const connRes = await composio.connections();
            const slugs = (connRes.connections || []).map((c: any) => (c.slug || c.app || "").toLowerCase());
            if (slugs.includes(app.toLowerCase())) {
              clearInterval(poll);
              setConnected(true);
              setConnecting(false);
              if (onConnect) onConnect(app);
            }
          } catch {}
        }, 3000);
        // Stop polling after 60s
        setTimeout(() => clearInterval(poll), 60000);
      } else if (res.requires_key) {
        // Redirect to integrations page for API key input
        window.open("/integrations?search=" + encodeURIComponent(app), "_blank");
      } else if (res.success) {
        setConnected(true);
        if (onConnect) onConnect(app);
      } else {
        alert(res.error || "Connection failed");
      }
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
    setConnecting(false);
  }

  return (
    <div className={`rounded-sm border ${connected ? "border-success/30 bg-success/5" : "border-amber-400/30 bg-amber-400/5"} p-3 animate-fade-in my-1`}>
      <div className="flex items-center gap-3">
        <img src={logoUrl} alt="" className="h-6 w-6 rounded-sm" onError={e => (e.currentTarget.style.display = "none")} />
        <div className="flex-1 min-w-0">
          <p className="text-[11px] font-semibold text-foreground">{appName || app}</p>
          <p className="text-[9px] text-muted">
            {connected ? "Connected! The agent will continue automatically." : "This app needs to be connected before the agent can use it."}
          </p>
        </div>
        {connected ? (
          <span className="flex items-center gap-1 text-[9px] text-success flex-shrink-0"><CheckCircle2 className="h-3.5 w-3.5" /> Ready</span>
        ) : (
          <button onClick={handleConnect} disabled={connecting}
            className="flex items-center gap-1.5 rounded-sm bg-primary px-3 py-1.5 text-[9px] font-medium text-white hover:bg-primary-hover disabled:opacity-50 flex-shrink-0">
            {connecting ? <Loader2 className="h-3 w-3 animate-spin" /> : <ExternalLink className="h-3 w-3" />}
            {connecting ? "Connecting..." : "Connect Now"}
          </button>
        )}
      </div>
    </div>
  );
}
