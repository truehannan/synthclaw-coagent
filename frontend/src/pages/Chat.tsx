import { useState, useRef, useEffect, Component, type ReactNode, type ErrorInfo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Send, Square, Trash2, ChevronDown, ChevronRight, Plus, Brain, Puzzle } from "lucide-react";
import { chat, models, sessions, society, providers as providersApi, skills as skillsApi, getToken } from "@/lib/api";
import type { Message } from "@/lib/types";
import ReactMarkdown from "react-markdown";
import Mascot from "@/components/Mascot";
import { ChatEvent, ThinkingIndicator, ToolCallCard, AgentBadge, PlanCard, ProgressCard, ErrorCard } from "@/components/ChatEvents";

// Error Boundary to prevent entire chat from crashing
class ChatItemBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };
  static getDerivedStateFromError(error: Error) { return { error: error.message }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error("ChatItem render error:", error, info); }
  render() {
    if (this.state.error) {
      return <div className="text-[9px] text-danger/60 px-2 py-1">[Render error: {this.state.error}]</div>;
    }
    return this.props.children;
  }
}

interface ProviderModels { name: string; emoji: string; models: string[]; expanded: boolean; }
interface ChatItem { type: "user" | "assistant" | "event"; content?: string; event?: ChatEvent; }

export default function Chat() {
  const { id: sessionId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [items, setItems] = useState<ChatItem[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [liveEvents, setLiveEvents] = useState<ChatEvent[]>([]);
  const liveEventsRef = useRef<ChatEvent[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Model switcher
  const [currentModel, setCurrentModel] = useState("");
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [providerModels, setProviderModels] = useState<ProviderModels[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelSearch, setModelSearch] = useState("");

  // Skills picker
  const [showSkillsPicker, setShowSkillsPicker] = useState(false);
  const [skillsList, setSkillsList] = useState<any[]>([]);
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [skillsSearch, setSkillsSearch] = useState("");

  // Society panel
  const [showSociety, setShowSociety] = useState(false);
  const [societyData, setSocietyData] = useState<any>(null);
  const societyPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (sessionId) {
      // Switch backend to this session, then load its history
      sessions.switch(sessionId).then(() => {
        loadHistory();
      }).catch(() => loadHistory());
    } else {
      // New chat — clear items, don't load history
      setItems([]);
    }
    models.current().then(r => setCurrentModel(r.model)).catch(() => {});
    return () => {
      if (societyPollRef.current) { clearInterval(societyPollRef.current); societyPollRef.current = null; }
    };
  }, [sessionId]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [items, streamText, liveEvents]);

  async function loadHistory() {
    try {
      const res = await chat.history();
      const msgs = (res.messages || []).map((m: Message) => ({ type: m.role as "user" | "assistant", content: m.content }));
      setItems(msgs);
    } catch {}
  }

  async function handleSend() {
    if (!input.trim() || streaming) return;
    const userMsg = input.trim();
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";

    // Create session on first message
    let newSessionId = sessionId;
    if (!sessionId && items.length === 0) {
      try {
        const name = userMsg.slice(0, 30) + (userMsg.length > 30 ? "..." : "");
        const res = await sessions.create(name);
        if (res.session?.id) {
          await sessions.switch(res.session.id);
          newSessionId = res.session.id;
        }
      } catch {}
    }

    setItems(prev => [...prev, { type: "user", content: userMsg }]);
    setStreaming(true);
    setStreamText("");
    setLiveEvents([]);
    liveEventsRef.current = [];

    try {
      // Use the full agentic endpoint
      const res = await fetch("/api/chat/run", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Token": getToken() },
        body: JSON.stringify({ message: userMsg, model: currentModel || undefined }),
      });

      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(errBody || `HTTP ${res.status}`);
      }
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let finalResponse = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event: ChatEvent = JSON.parse(line.slice(6));

            if (event.type === "done") {
              finalResponse = event.full || event.content || "";
            } else if (event.type === "error") {
              finalResponse = `Error: ${event.message || event.content || "Unknown error"}`;
            } else {
              // All other events go to live display
              liveEventsRef.current = [...liveEventsRef.current, event];
              setLiveEvents([...liveEventsRef.current]);
            }
          } catch {}
        }
      }

      // Commit final response — use ref (not stale state)
      const eventsToCommit = liveEventsRef.current
        .filter(e => e.type !== "thinking")
        .map(e => ({ type: "event" as const, event: e }));

      if (finalResponse) {
        setItems(prev => [...prev, ...eventsToCommit, { type: "assistant", content: finalResponse }]);
      } else if (eventsToCommit.length > 0) {
        setItems(prev => [...prev, ...eventsToCommit]);
      } else {
        setItems(prev => [...prev, { type: "assistant", content: "No response received. Check /api/chat/debug for configuration." }]);
      }
    } catch (err: any) {
      setItems(prev => [...prev, { type: "assistant", content: `Error: ${err.message}` }]);
    } finally {
      setStreaming(false);
      setStreamText("");
      setLiveEvents([]);
      if (newSessionId && newSessionId !== sessionId) {
        navigate(`/chat/${newSessionId}`, { replace: true });
      }
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  async function handleStop() { await chat.stop(); setStreaming(false); }
  async function handleClear() { await chat.clear(); setItems([]); }

  async function switchModel(model: string) {
    await models.switch(model);
    setCurrentModel(model);
    setShowModelPicker(false);
  }

  async function openModelPicker() {
    if (showModelPicker) { setShowModelPicker(false); return; }
    setShowModelPicker(true);
    if (providerModels.length === 0) {
      setLoadingModels(true);
      try {
        const provRes = await providersApi.list();
        const grouped: ProviderModels[] = [];
        for (const p of (provRes.providers || [])) {
          try {
            const mRes = await providersApi.models(p.name);
            if (mRes.models?.length > 0) grouped.push({ name: p.name, emoji: p.emoji, models: mRes.models, expanded: false });
          } catch {}
        }
        setProviderModels(grouped);
      } catch {}
      setLoadingModels(false);
    }
  }

  function toggleProviderExpand(idx: number) {
    setProviderModels(prev => prev.map((p, i) => i === idx ? { ...p, expanded: !p.expanded } : { ...p, expanded: false }));
  }

  async function openSkillsPicker() {
    if (showSkillsPicker) { setShowSkillsPicker(false); return; }
    setShowSkillsPicker(true);
    if (skillsList.length === 0) {
      try { const res = await skillsApi.list(); setSkillsList(res.skills || []); } catch {}
    }
  }

  function toggleSkill(name: string) { setSelectedSkills(prev => prev.includes(name) ? prev.filter(s => s !== name) : [...prev, name]); }
  async function toggleSociety() {
    if (!showSociety) {
      try { setSocietyData(await society.status()); } catch {}
      // Start polling society status while panel is open
      if (!societyPollRef.current) {
        societyPollRef.current = setInterval(async () => {
          try { setSocietyData(await society.status()); } catch {}
        }, 2000);
      }
    } else {
      // Stop polling when closing
      if (societyPollRef.current) { clearInterval(societyPollRef.current); societyPollRef.current = null; }
    }
    setShowSociety(!showSociety);
  }
  function handleNewChat() { navigate("/chat"); setItems([]); setLiveEvents([]); }

  return (
    <div className="flex h-full">
      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-h-0 min-w-0">
        {/* Top bar */}
        <div className="flex items-center justify-between border-b border-border bg-card px-4 py-2">
          <div className="flex items-center gap-2">
            <button onClick={handleNewChat} title="New Chat" className="rounded-sm border border-border p-1.5 text-muted hover:border-primary hover:text-primary">
              <Plus className="h-3.5 w-3.5" />
            </button>
            <span className="text-[10px] text-muted truncate max-w-[200px]">{currentModel || "No model"}</span>
          </div>
          <button onClick={toggleSociety} title="Agent Society"
            className={`flex items-center gap-1.5 rounded-sm border px-2.5 py-1.5 text-[10px] font-medium ${showSociety ? "border-primary text-primary bg-primary/5" : "border-border text-muted hover:border-primary hover:text-primary"}`}>
            <Brain className="h-3.5 w-3.5" />
            <span>Agents</span>
          </button>
        </div>

        {/* Messages + Events */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto max-w-3xl space-y-3">
            {items.length === 0 && !streaming && (
              <div className="flex h-64 items-center justify-center text-center">
                <div>
                  <Mascot className="mx-auto mb-4" />
                  <p className="text-lg font-semibold text-primary">[+] Conclave</p>
                  <p className="mt-2 text-xs text-muted">Send a message to start. Full agentic loop active.</p>
                </div>
              </div>
            )}

            {items.map((item, i) => {
              if (item.type === "user") {
                return (
                  <ChatItemBoundary key={i}>
                    <div className="flex justify-end animate-fade-in">
                      <div className="max-w-[85%] rounded-sm bg-primary-dim px-4 py-3 text-sm">
                        <p className="whitespace-pre-wrap">{item.content}</p>
                      </div>
                    </div>
                  </ChatItemBoundary>
                );
              }
              if (item.type === "assistant") {
                return (
                  <ChatItemBoundary key={i}>
                    <div className="animate-fade-in">
                      <div className="max-w-[85%] rounded-sm border border-border bg-card px-4 py-3 text-sm">
                        <div className="prose prose-invert prose-sm max-w-none">
                          <ReactMarkdown>{item.content || ""}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  </ChatItemBoundary>
                );
              }
              if (item.type === "event" && item.event) {
                return <ChatItemBoundary key={i}>{renderEvent(item.event)}</ChatItemBoundary>;
              }
              return null;
            })}

            {/* Live events while streaming */}
            {streaming && (
              <div className="space-y-1">
                {liveEvents.map((event, i) => (
                  <ChatItemBoundary key={i}>{renderEvent(event)}</ChatItemBoundary>
                ))}
                {liveEvents.length === 0 && <ThinkingIndicator />}
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input area */}
        <div className="border-t border-border bg-card px-4 py-3">
          <div className="mx-auto max-w-3xl">
            {/* Model + Skills row */}
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              {/* Model picker */}
              <div className="relative">
                <button onClick={openModelPicker}
                  className="flex items-center gap-1.5 rounded-sm border border-border px-2 py-1 text-[9px] text-muted hover:border-primary hover:text-primary">
                  <span className="max-w-[150px] truncate">{currentModel || "Model"}</span>
                  <ChevronDown className="h-2.5 w-2.5" />
                </button>
                {showModelPicker && (
                  <div className="absolute left-0 bottom-full z-50 mb-1 w-80 max-h-72 overflow-hidden rounded-sm border border-border bg-card shadow-lg flex flex-col">
                    <div className="p-2 border-b border-border">
                      <input value={modelSearch} onChange={e => setModelSearch(e.target.value)} placeholder="Search..."
                        autoFocus className="w-full rounded-sm border border-border bg-background px-2 py-1 text-[10px] outline-none focus:border-primary" />
                    </div>
                    <div className="overflow-y-auto flex-1">
                      {loadingModels && <p className="px-3 py-3 text-[10px] text-muted animate-pulse">Fetching live models...</p>}
                      {providerModels.map((p, idx) => {
                        const filtered = modelSearch ? p.models.filter(m => m.toLowerCase().includes(modelSearch.toLowerCase())) : p.models;
                        if (modelSearch && filtered.length === 0) return null;
                        return (
                          <div key={p.name}>
                            <button onClick={() => toggleProviderExpand(idx)}
                              className="flex w-full items-center gap-2 px-3 py-1.5 text-left hover:bg-card-hover border-b border-border/30">
                              {p.expanded ? <ChevronDown className="h-2.5 w-2.5 text-muted" /> : <ChevronRight className="h-2.5 w-2.5 text-muted" />}
                              <span className="text-xs">{p.emoji}</span>
                              <span className="text-[10px] font-semibold">{p.name}</span>
                              <span className="ml-auto text-[9px] text-muted">{filtered.length}</span>
                            </button>
                            {(p.expanded || !!modelSearch) && filtered.map(m => (
                              <button key={m} onClick={() => switchModel(m)}
                                className={`block w-full px-5 py-1 text-left text-[10px] hover:bg-card-hover ${m === currentModel ? "text-primary" : "text-muted"}`}>{m}</button>
                            ))}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>

              {/* Skills picker */}
              <div className="relative">
                <button onClick={openSkillsPicker}
                  className="flex items-center gap-1.5 rounded-sm border border-border px-2 py-1 text-[9px] text-muted hover:border-primary hover:text-primary">
                  <Puzzle className="h-2.5 w-2.5" />
                  <span>Skills{selectedSkills.length > 0 ? ` (${selectedSkills.length})` : ""}</span>
                </button>
                {showSkillsPicker && (
                  <div className="absolute left-0 bottom-full z-50 mb-1 w-56 max-h-48 overflow-hidden rounded-sm border border-border bg-card shadow-lg flex flex-col">
                    <div className="p-2 border-b border-border">
                      <input value={skillsSearch} onChange={e => setSkillsSearch(e.target.value)} placeholder="Search..."
                        autoFocus className="w-full rounded-sm border border-border bg-background px-2 py-1 text-[10px] outline-none focus:border-primary" />
                    </div>
                    <div className="overflow-y-auto flex-1 p-1">
                      {skillsList.filter(s => !skillsSearch || (s.name || s).toLowerCase().includes(skillsSearch.toLowerCase())).map((s: any, i: number) => {
                        const name = s.name || s;
                        const active = selectedSkills.includes(name);
                        return (
                          <button key={i} onClick={() => toggleSkill(name)}
                            className={`flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-left text-[10px] ${active ? "bg-primary/10 text-primary" : "text-muted hover:bg-card-hover"}`}>
                            <span className={`h-2 w-2 rounded-full border ${active ? "bg-primary border-primary" : "border-muted"}`} />
                            <span className="truncate">{name}</span>
                          </button>
                        );
                      })}
                      {skillsList.length === 0 && <p className="px-2 py-2 text-[9px] text-muted text-center">No skills</p>}
                    </div>
                  </div>
                )}
              </div>

              {/* Selected skills tags */}
              {selectedSkills.map(s => (
                <span key={s} onClick={() => toggleSkill(s)}
                  className="rounded-full bg-primary/10 px-2 py-0.5 text-[8px] text-primary cursor-pointer hover:bg-primary/20">{s} x</span>
              ))}
            </div>

            {/* Input row */}
            <div className="flex items-end gap-2">
              {items.length > 0 && (
                <button onClick={handleClear} title="Clear" className="rounded-sm border border-border p-2.5 text-muted hover:border-danger hover:text-danger">
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
              <textarea ref={inputRef} value={input}
                onChange={(e) => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px"; }}
                onKeyDown={handleKeyDown} placeholder="Message Conclave..." rows={1}
                className="flex-1 resize-none rounded-sm border border-border bg-background px-3 py-2.5 text-sm text-foreground placeholder-muted/50 outline-none focus:border-primary"
                style={{ maxHeight: "160px", overflow: "auto" }} />
              {streaming ? (
                <button onClick={handleStop} className="rounded-sm bg-danger/20 p-2.5 text-danger hover:bg-danger/30"><Square className="h-4 w-4" /></button>
              ) : (
                <button onClick={handleSend} disabled={!input.trim()} className="rounded-sm bg-primary p-2.5 text-white hover:bg-primary-hover disabled:opacity-30"><Send className="h-4 w-4" /></button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Agent Society — Right Sidebar with Thread-Style */}
      {showSociety && (
        <div className="w-72 border-l border-border bg-card flex flex-col h-full overflow-hidden flex-shrink-0">
          {/* Sidebar header */}
          <div className="flex items-center justify-between px-3 py-2.5 border-b border-border">
            <div className="flex items-center gap-2">
              <Brain className="h-3.5 w-3.5 text-primary" />
              <span className="text-[10px] font-bold uppercase tracking-wider text-foreground">Agents</span>
            </div>
            <button onClick={() => setShowSociety(false)} className="text-[9px] text-muted hover:text-foreground rounded-sm border border-border px-1.5 py-0.5">x</button>
          </div>

          {/* Status bar */}
          {societyData?.agents && (
            <div className="px-3 py-1.5 border-b border-border bg-background/50">
              <div className="flex items-center gap-3 text-[8px] text-muted">
                <span className="flex items-center gap-1">
                  <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                  {societyData.agents.active || 0} active
                </span>
                <span>{societyData.agents.total || 0} total</span>
              </div>
            </div>
          )}

          {/* Thread-style agent list */}
          <div className="flex-1 overflow-y-auto px-2 py-2">
            {societyData?.active?.length > 0 ? (
              <div className="relative">
                {/* Main thread line */}
                <div className="absolute left-[9px] top-2 bottom-2 w-px bg-border" />

                {societyData.active.map((a: any, i: number) => {
                  const colorMap: Record<string, string> = {
                    orchestrator: "#ef4444", researcher: "#3b82f6", executor: "#10b981",
                    reviewer: "#f59e0b", observer: "#8b5cf6", planner: "#ec4899", coder: "#14b8a6",
                  };
                  const color = colorMap[a.role?.toLowerCase()] || "#6b7280";
                  const hasChildren = a.subtasks?.length > 0 || a.children?.length > 0;

                  return (
                    <div key={i} className="relative mb-0.5">
                      {/* Thread node */}
                      <div className="flex items-start gap-2 pl-0 py-1.5">
                        {/* Branch dot */}
                        <div className="relative flex-shrink-0 mt-0.5 z-10">
                          <div className="h-[18px] w-[18px] rounded-full border-2 flex items-center justify-center bg-card"
                            style={{ borderColor: color }}>
                            <div className="h-1.5 w-1.5 rounded-full" style={{ background: color, opacity: a.status === "executing" ? 1 : 0.5 }} />
                          </div>
                        </div>
                        {/* Agent info */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1">
                            <span className="text-[9px] font-bold capitalize truncate" style={{ color }}>
                              {a.name || a.role}
                            </span>
                            <span className={`rounded-full px-1 py-px text-[7px] font-medium leading-none ${
                              a.status === "executing" ? "bg-success/15 text-success" :
                              a.status === "thinking" ? "bg-blue-400/15 text-blue-400" :
                              a.status === "waiting" ? "bg-amber-400/15 text-amber-400" :
                              "bg-muted/10 text-muted"
                            }`}>{a.status || "idle"}</span>
                          </div>
                          <p className="text-[8px] text-muted mt-0.5 line-clamp-2">{a.task || "Awaiting"}</p>
                          {a.elapsed && <span className="text-[7px] text-muted/50">{a.elapsed}s</span>}
                        </div>
                      </div>

                      {/* Sub-tasks / children branch */}
                      {(a.subtasks || a.children || []).map((child: any, ci: number) => (
                        <div key={ci} className="relative ml-4 pl-3 py-0.5 border-l" style={{ borderColor: `${color}40` }}>
                          <div className="absolute left-0 top-[8px] w-2 h-px" style={{ background: color, opacity: 0.4 }} />
                          <div className="flex items-center gap-1.5">
                            <div className="h-1 w-1 rounded-full flex-shrink-0" style={{ background: color, opacity: 0.6 }} />
                            <span className="text-[8px] text-muted truncate">{child.task || child.name || child}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex h-full items-center justify-center">
                <div className="text-center py-8">
                  <Brain className="h-6 w-6 text-muted/20 mx-auto mb-2" />
                  <p className="text-[9px] text-muted font-medium">No active agents</p>
                  <p className="text-[8px] text-muted/50 mt-1 max-w-[160px] mx-auto">
                    Thread view shows agents when processing tasks
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar footer */}
          <div className="border-t border-border px-3 py-1.5">
            <p className="text-[7px] text-muted/40 text-center">Real-time orchestration</p>
          </div>
        </div>
      )}
    </div>
  );
}

function renderEvent(event: ChatEvent) {
  if (!event || !event.type) return null;
  try {
    switch (event.type) {
      case "thinking":
        return <ThinkingIndicator content={event.content} />;
      case "progress":
        return <ProgressCard content={event.content || ""} />;
      case "agent_step":
        return <AgentBadge role="executor" task={event.content?.replace("  ──•", "").trim()} />;
      case "agent_spawn":
        return <AgentBadge role={event.agent?.role || "agent"} task={event.agent?.task} />;
      case "tool_call":
        return <ToolCallCard tool={event.tool || "unknown"} args={event.args} />;
      case "tool_result":
        return <ToolCallCard tool={event.tool || "tool"} output={event.output} collapsed={false} />;
      case "plan":
        return <PlanCard steps={Array.isArray(event.steps) ? event.steps : []} />;
      case "text":
        return (
          <div className="text-[10px] text-muted border-l-2 border-primary/30 pl-2 py-0.5 animate-fade-in">
            {event.content || ""}
          </div>
        );
      case "error":
        return <ErrorCard message={event.message || event.content || "Unknown error"} />;
      default:
        // Unknown event type — show raw content as text
        return event.content ? (
          <div className="text-[10px] text-muted pl-2 py-0.5">{event.content}</div>
        ) : null;
    }
  } catch (e) {
    return <div className="text-[9px] text-danger/60 px-2 py-1">[Event render error]</div>;
  }
}
