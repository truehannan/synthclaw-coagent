import { useEffect, useState } from "react";
import { skills as api } from "@/lib/api";
import { Download, Trash2, RefreshCw } from "lucide-react";

export default function Skills() {
  const [skillsList, setSkillsList] = useState<any[]>([]);
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => { load(); }, []);

  async function load() {
    try {
      const res = await api.list();
      setSkillsList(res.skills || []);
    } catch { setSkillsList([]); }
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
      <div className="space-y-1">
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
    </div>
  );
}
