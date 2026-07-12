import { useEffect, useState } from "react";
import { sessions as api } from "@/lib/api";
import { Plus, Trash2, Copy, Check } from "lucide-react";
import type { Session } from "@/lib/types";

export default function SessionSidebar() {
  const [list, setList] = useState<Session[]>([]);
  const [active, setActive] = useState("default");
  const [copied, setCopied] = useState("");

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      const res = await api.list();
      setList(res.sessions || []);
      setActive(res.active || "default");
    } catch {}
  }

  async function create() {
    const name = `Session ${list.length + 1}`;
    await api.create(name);
    load();
  }

  async function switchTo(id: string) {
    await api.switch(id);
    setActive(id);
  }

  async function remove(id: string) {
    if (id === active) return;
    await api.del(id);
    load();
  }

  function copyId(id: string) {
    navigator.clipboard.writeText(id);
    setCopied(id);
    setTimeout(() => setCopied(""), 1500);
  }

  return (
    <div className="border-t border-border px-2 py-2">
      <div className="flex items-center justify-between px-2 py-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted">Sessions</span>
        <button onClick={create} className="text-muted hover:text-primary"><Plus className="h-3 w-3" /></button>
      </div>
      <div className="mt-1 max-h-32 overflow-y-auto space-y-0.5">
        {list.map((s) => (
          <div key={s.id} className={`group flex items-center gap-1.5 rounded-sm px-2 py-1 text-[10px] cursor-pointer ${s.id === active ? "bg-primary-dim text-primary" : "text-muted hover:bg-card-hover hover:text-foreground"}`}
            onClick={() => switchTo(s.id)}>
            <span className="flex-1 truncate">{s.name}</span>
            <button onClick={(e) => { e.stopPropagation(); copyId(s.id); }} className="opacity-0 group-hover:opacity-100 text-muted hover:text-foreground">
              {copied === s.id ? <Check className="h-2.5 w-2.5 text-success" /> : <Copy className="h-2.5 w-2.5" />}
            </button>
            {s.id !== active && (
              <button onClick={(e) => { e.stopPropagation(); remove(s.id); }} className="opacity-0 group-hover:opacity-100 text-muted hover:text-danger">
                <Trash2 className="h-2.5 w-2.5" />
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
