import { useState, useRef, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Send, Square, CheckCircle, XCircle, Trash2, ChevronDown, ChevronRight, Plus, Brain } from "lucide-react";
import { chat, models, sessions, society, providers as providersApi } from "@/lib/api";
import type { Message } from "@/lib/types";
import ReactMarkdown from "react-markdown";
import Mascot from "@/components/Mascot";

interface ProviderModels {
  name: string;
  emoji: string;
  models: string[];
  expanded: boolean;
}

export default function Chat() {
  const { id: sessionId } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [pendingApproval, setPendingApproval] = useState(false);
  const [approvalDesc, setApprovalDesc] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Model switcher
  const [currentModel, setCurrentModel] = useState("");
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [providerModels, setProviderModels] = useState<ProviderModels[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelSearch, setModelSearch] = useState("");

  // Agent society sidebar
  const [showSociety, setShowSociety] = useState(false);
  const [societyData, setSocietyData] = useState<any>(null);

  // Load chat when session changes
  useEffect(() => {
    // If we have a sessionId, switch to it on the backend
    if (sessionId) {
      sessions.switch(sessionId).then(() => loadHistory()).catch(() => loadHistory());
    } else {
      loadHistory();
    }
    models.current().then(r => setCurrentModel(r.model)).catch(() => {});
    pollRef.current = setInterval(pollTaskStatus, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [sessionId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamText]);

  async function pollTaskStatus() {
    try {
      const res = await chat.taskStatus();
      if (res.pending_approval?.active && !res.pending_approval?.resolved) {
        setPendingApproval(true);
        setApprovalDesc(res.pending_approval.description || "");
      } else {
        setPendingApproval(false);
        setApprovalDesc("");
      }
    } catch {}
  }

  async function loadHistory() {
    try {
      const res = await chat.history();
      setMessages(res.messages || []);
    } catch {}
  }

  async function handleSend() {
    if (!input.trim() || streaming) return;
    const userMsg = input.trim();
    setInput("");
    if (inputRef.current) inputRef.current.style.height = "auto";

    // If no session yet, create one named after first message
    if (!sessionId && messages.length === 0) {
      try {
        const name = userMsg.slice(0, 30) + (userMsg.length > 30 ? "..." : "");
        const res = await sessions.create(name);
        if (res.session?.id) {
          // Switch backend to new session
          await sessions.switch(res.session.id);
          navigate(`/chat/${res.session.id}`, { replace: true });
        }
      } catch {}
    }

    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setStreaming(true);
    setStreamText("");

    try {
      const res = await chat.sendStream(userMsg, currentModel || undefined);
      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.token) { fullText += data.token; setStreamText(fullText); }
            if (data.done) { setMessages(prev => [...prev, { role: "assistant", content: data.full || fullText }]); setStreamText(""); }
            if (data.error) { setMessages(prev => [...prev, { role: "assistant", content: `Error: ${data.error}` }]); setStreamText(""); }
          } catch {}
        }
      }
    } catch (err: any) {
      setMessages(prev => [...prev, { role: "assistant", content: `Error: ${err.message}` }]);
    } finally {
      setStreaming(false);
      setStreamText("");
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  }

  async function handleStop() { await chat.stop(); setStreaming(false); }
  async function handleClear() { await chat.clear(); setMessages([]); }

  async function switchModel(model: string) {
    await models.switch(model);
    setCurrentModel(model);
    setShowModelPicker(false);
  }

  async function openModelPicker() {
    if (showModelPicker) { setShowModelPicker(false); return; }
    if (providerModels.length === 0) {
      setLoadingModels(true);
      try {
        // Fetch providers list then fetch models for each configured one
        const provRes = await providersApi.list();
        const provList = provRes.providers || [];
        const grouped: ProviderModels[] = [];
        for (const p of provList) {
          try {
            const mRes = await providersApi.models(p.name);
            if (mRes.models && mRes.models.length > 0) {
              grouped.push({ name: p.name, emoji: p.emoji, models: mRes.models, expanded: false });
            }
          } catch {}
        }
        setProviderModels(grouped);
      } catch {}
      setLoadingModels(false);
    }
    setShowModelPicker(true);
  }

  function toggleProviderExpand(idx: number) {
    setProviderModels(prev => prev.map((p, i) => i === idx ? { ...p, expanded: !p.expanded } : { ...p, expanded: false }));
  }

  async function toggleSociety() {
    if (!showSociety) {
      try { const res = await society.status(); setSocietyData(res); } catch {}
    }
    setShowSociety(!showSociety);
  }

  function handleNewChat() {
    navigate("/chat");
    setMessages([]);
  }

  return (
    <div className="flex h-full">
      <div className="flex flex-1 flex-col">
        {/* Top bar */}
        <div className="flex items-center justify-between border-b border-border bg-card px-4 py-2">
          <div className="flex items-center gap-2">
            <button onClick={handleNewChat} title="New Chat"
              className="rounded-sm border border-border p-1.5 text-muted hover:border-primary hover:text-primary">
              <Plus className="h-3.5 w-3.5" />
            </button>
            <span className="text-[10px] text-muted truncate max-w-[200px]">{currentModel || "No model"}</span>
          </div>
          <button onClick={toggleSociety} title="Agent Society"
            className={`rounded-sm border p-1.5 ${showSociety ? "border-primary text-primary" : "border-border text-muted hover:border-primary hover:text-primary"}`}>
            <Brain className="h-3.5 w-3.5" />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto max-w-3xl space-y-4">
            {messages.length === 0 && !streaming && (
              <div className="flex h-64 items-center justify-center text-center">
                <div>
                  <Mascot className="mx-auto mb-4" />
                  <p className="text-lg font-semibold text-primary">[+] SynthClaw</p>
                  <p className="mt-2 text-xs text-muted">Send a message to start. Your agent is ready.</p>
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <div key={i} className={`animate-fade-in ${msg.role === "user" ? "flex justify-end" : ""}`}>
                <div className={`max-w-[85%] rounded-sm px-4 py-3 text-sm leading-relaxed ${
                  msg.role === "user" ? "bg-primary-dim text-foreground" : "border border-border bg-card"
                }`}>
                  {msg.role === "assistant" ? (
                    <div className="prose prose-invert prose-sm max-w-none"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                  ) : (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  )}
                </div>
              </div>
            ))}
            {streaming && streamText && (
              <div className="animate-fade-in border border-border bg-card rounded-sm px-4 py-3 text-sm max-w-[85%]">
                <div className="prose prose-invert prose-sm max-w-none"><ReactMarkdown>{streamText}</ReactMarkdown></div>
                <span className="inline-block h-3 w-1 animate-pulse bg-primary ml-0.5" />
              </div>
            )}
            {streaming && !streamText && (
              <div className="flex items-center gap-2 text-xs text-muted animate-fade-in">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" /> Thinking...
              </div>
            )}
            {pendingApproval && (
              <div className="animate-slide-up rounded-sm border border-warning bg-warning/10 p-4">
                <p className="text-sm font-semibold text-warning">Approval Required</p>
                <p className="mt-1 text-xs text-muted">{approvalDesc || "Dangerous operation pending."}</p>
                <div className="mt-3 flex gap-2">
                  <button onClick={() => { chat.approve(); setPendingApproval(false); }}
                    className="flex items-center gap-1.5 rounded-sm bg-success/20 px-3 py-1.5 text-xs font-medium text-success hover:bg-success/30">
                    <CheckCircle className="h-3.5 w-3.5" /> Approve
                  </button>
                  <button onClick={() => { chat.deny(); setPendingApproval(false); }}
                    className="flex items-center gap-1.5 rounded-sm bg-danger/20 px-3 py-1.5 text-xs font-medium text-danger hover:bg-danger/30">
                    <XCircle className="h-3.5 w-3.5" /> Deny
                  </button>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-border bg-card px-4 py-3">
          <div className="mx-auto max-w-3xl">
            {/* Model selector row */}
            <div className="flex items-center gap-2 mb-2">
              <div className="relative">
                <button onClick={openModelPicker}
                  className="flex items-center gap-1.5 rounded-sm border border-border px-2 py-1 text-[9px] text-muted hover:border-primary hover:text-primary">
                  <span className="max-w-[150px] truncate">{currentModel || "Select model"}</span>
                  <ChevronDown className="h-2.5 w-2.5" />
                </button>
                {showModelPicker && (
                  <div className="absolute left-0 bottom-full z-50 mb-1 w-80 max-h-72 overflow-hidden rounded-sm border border-border bg-card shadow-lg flex flex-col">
                    {/* Search */}
                    <div className="p-2 border-b border-border">
                      <input value={modelSearch} onChange={e => setModelSearch(e.target.value)} placeholder="Search models..."
                        autoFocus className="w-full rounded-sm border border-border bg-background px-2 py-1 text-[10px] outline-none focus:border-primary" />
                    </div>
                    <div className="overflow-y-auto flex-1">
                      {loadingModels && <p className="px-3 py-3 text-[10px] text-muted animate-pulse">Fetching live models from providers...</p>}
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
                            {(p.expanded || !!modelSearch) && (
                              <div className="bg-background/50">
                                {filtered.map(m => (
                                  <button key={m} onClick={() => switchModel(m)}
                                    className={`block w-full px-5 py-1 text-left text-[10px] hover:bg-card-hover ${m === currentModel ? "text-primary" : "text-muted"}`}>
                                    {m}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </div>
            {/* Input row */}
            <div className="flex items-end gap-2">
              {messages.length > 0 && (
                <button onClick={handleClear} title="Clear" className="rounded-sm border border-border p-2.5 text-muted hover:border-danger hover:text-danger">
                  <Trash2 className="h-4 w-4" />
                </button>
              )}
              <textarea ref={inputRef} value={input}
                onChange={(e) => { setInput(e.target.value); e.target.style.height = "auto"; e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px"; }}
                onKeyDown={handleKeyDown} placeholder="Message SynthClaw... (type / for commands)" rows={1}
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

      {/* Agent Society right sidebar */}
      {showSociety && (
        <div className="w-64 border-l border-border bg-card overflow-y-auto p-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted mb-3">Agent Society</p>
          {societyData?.active && societyData.active.length > 0 ? (
            <div className="space-y-2">
              {societyData.active.map((a: any, i: number) => (
                <div key={i} className="rounded-sm border border-border p-2">
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                    <span className="text-[10px] font-semibold">{a.name || a.role}</span>
                  </div>
                  <p className="mt-1 text-[9px] text-muted truncate">{a.task || a.status}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Brain className="h-6 w-6 text-muted mx-auto mb-2" />
              <p className="text-[10px] text-muted">No active agents</p>
              <p className="text-[9px] text-muted mt-1">Agents spawn for complex tasks</p>
            </div>
          )}
          {societyData?.agents && (
            <div className="mt-4 border-t border-border pt-3 space-y-1 text-[9px] text-muted">
              <div className="flex justify-between"><span>Active</span><span className="text-foreground">{societyData.agents.active || 0}</span></div>
              <div className="flex justify-between"><span>Completed</span><span className="text-foreground">{societyData.agents.total_completed || 0}</span></div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
