import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { sessions as api } from "@/lib/api";
import { Plus, Trash2 } from "lucide-react";
import type { Session } from "@/lib/types";

export default function SessionSidebar() {
  const [list, setList] = useState<Session[]>([]);
  const [active, setActive] = useState("default");
  const navigate = useNavigate();

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      const res = await api.list();
      setList(res.sessions || []);
      setActive(res.active || "default");
    } catch {}
  }

  async function switchTo(id: string) {
    await api.switch(id);
    setActive(id);
    navigate(`/chat/${id}`);
  }

  async function remove(id: string) {
    if (id === active) return;
    await api.del(id);
    load();
  }

  function handleNewChat() {
    navigate("/chat");
  }

  return (
    <div className="border-t border-border px-2 py-2">
      <div className="flex items-center justify-between px-2 py-1">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted">Chats</span>
        <button onClick={handleNewChat} title="New Chat" className="text-muted hover:text-primary"><Plus className="h-3 w-3" /></button>
      </div>
      <div className="mt-1 max-h-40 overflow-y-auto space-y-0.5">
        {list.map((s) => (
          <div key={s.id}
            className={`group flex items-center gap-1.5 rounded-sm px-2 py-1.5 text-[10px] cursor-pointer ${
              s.id === active ? "bg-primary-dim text-primary" : "text-muted hover:bg-card-hover hover:text-foreground"
            }`}
            onClick={() => switchTo(s.id)}>
            <span className="flex-1 truncate">{s.name}</span>
            {s.id !== active && (
              <button onClick={(e) => { e.stopPropagation(); remove(s.id); }}
                className="opacity-0 group-hover:opacity-100 text-muted hover:text-danger">
                <Trash2 className="h-2.5 w-2.5" />
              </button>
            )}
          </div>
        ))}
        {list.length === 0 && (
          <p className="px-2 py-2 text-[9px] text-muted">No sessions yet</p>
        )}
      </div>
    </div>
  );
}
