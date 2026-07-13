import { useState, useRef, useEffect } from "react";
import { Send, Square, CheckCircle, XCircle } from "lucide-react";
import { chat } from "@/lib/api";
import type { Message } from "@/lib/types";
import ReactMarkdown from "react-markdown";
import Mascot from "@/components/Mascot";

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [pendingApproval, setPendingApproval] = useState(false);
  const [approvalDesc, setApprovalDesc] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    loadHistory();
    // Start polling for task status / approval requests
    pollRef.current = setInterval(pollTaskStatus, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

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
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);
    setStreaming(true);
    setStreamText("");

    try {
      const res = await chat.sendStream(userMsg);
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));
          if (data.token) {
            fullText += data.token;
            setStreamText(fullText);
          }
          if (data.done) {
            setMessages((prev) => [...prev, { role: "assistant", content: data.full || fullText }]);
            setStreamText("");
          }
          if (data.error) {
            setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${data.error}` }]);
            setStreamText("");
          }
        }
      }
    } catch (err: any) {
      setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${err.message}` }]);
    } finally {
      setStreaming(false);
      setStreamText("");
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  async function handleStop() {
    await chat.stop();
    setStreaming(false);
  }

  return (
    <div className="flex h-full flex-col">
      {/* Messages area */}
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
                msg.role === "user"
                  ? "bg-primary-dim text-foreground"
                  : "border border-border bg-card"
              }`}>
                {msg.role === "assistant" ? (
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {/* Streaming indicator */}
          {streaming && streamText && (
            <div className="animate-fade-in border border-border bg-card rounded-sm px-4 py-3 text-sm max-w-[85%]">
              <div className="prose prose-invert prose-sm max-w-none">
                <ReactMarkdown>{streamText}</ReactMarkdown>
              </div>
              <span className="inline-block h-3 w-1 animate-pulse bg-primary ml-0.5" />
            </div>
          )}

          {streaming && !streamText && (
            <div className="flex items-center gap-2 text-xs text-muted animate-fade-in">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-primary" />
              Thinking...
            </div>
          )}

          {/* Approval banner */}
          {pendingApproval && (
            <div className="animate-slide-up rounded-sm border border-warning bg-warning/10 p-4">
              <p className="text-sm font-semibold text-warning">Approval Required</p>
              <p className="mt-1 text-xs text-muted">
                {approvalDesc || "The agent wants to perform a dangerous operation."}
              </p>
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

      {/* Input area */}
      <div className="border-t border-border bg-card px-4 py-3">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message SynthClaw..."
            rows={1}
            className="flex-1 resize-none rounded-sm border border-border bg-background px-3 py-2.5 text-sm text-foreground placeholder-muted/50 outline-none focus:border-primary"
            style={{ maxHeight: "120px" }}
          />
          {streaming ? (
            <button onClick={handleStop}
              className="rounded-sm bg-danger/20 p-2.5 text-danger hover:bg-danger/30">
              <Square className="h-4 w-4" />
            </button>
          ) : (
            <button onClick={handleSend} disabled={!input.trim()}
              className="rounded-sm bg-primary p-2.5 text-white hover:bg-primary-hover disabled:opacity-30">
              <Send className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
