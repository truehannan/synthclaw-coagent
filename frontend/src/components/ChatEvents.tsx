import { useState } from "react";
import { ChevronDown, ChevronRight, Terminal, Brain, ListChecks, Loader2, AlertCircle, CheckCircle2, ExternalLink } from "lucide-react";

// ── Event Types ──────────────────────────────────────────────────────────────

export interface ChatEvent {
  type: "thinking" | "tool_call" | "tool_result" | "agent_step" | "agent_spawn" | "plan" | "text" | "progress" | "done" | "error";
  content?: string;
  full?: string;
  message?: string;
  tool?: string;
  args?: any;
  output?: string;
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

export function ToolCallCard({ tool, args, output, collapsed: initialCollapsed = true }: { tool: string; args?: any; output?: string; collapsed?: boolean }) {
  const [collapsed, setCollapsed] = useState(initialCollapsed);

  return (
    <div className="rounded-sm border border-border/60 bg-card/50 overflow-hidden animate-fade-in my-1">
      <button onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-card-hover">
        {collapsed ? <ChevronRight className="h-3 w-3 text-muted" /> : <ChevronDown className="h-3 w-3 text-muted" />}
        <Terminal className="h-3 w-3 text-primary" />
        <span className="text-[10px] font-semibold text-primary">{tool}</span>
        {output && <CheckCircle2 className="h-3 w-3 text-success ml-auto" />}
        {!output && <Loader2 className="h-3 w-3 animate-spin text-muted ml-auto" />}
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
