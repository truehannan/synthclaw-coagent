import { useEffect, useState } from "react";
import { providers as api, models as modelsApi } from "@/lib/api";
import type { Provider } from "@/lib/types";
import { Check, X, Key, RefreshCw } from "lucide-react";

export default function Providers() {
  const [list, setList] = useState<Provider[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [keyInput, setKeyInput] = useState("");
  const [modelsList, setModelsList] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentModel, setCurrentModel] = useState("");

  useEffect(() => {
    loadProviders();
    modelsApi.current().then(r => setCurrentModel(r.model)).catch(() => {});
  }, []);

  async function loadProviders() {
    const res = await api.list();
    setList(res.providers || []);
  }

  async function handleStoreKey() {
    if (!selected || !keyInput) return;
    await api.storeKey(selected, keyInput);
    setKeyInput("");
    loadProviders();
  }

  async function fetchModels(name: string) {
    setLoading(true);
    setSelected(name);
    try {
      const res = await api.models(name);
      setModelsList(res.models || []);
    } catch { setModelsList([]); }
    setLoading(false);
  }

  async function switchModel(model: string) {
    await modelsApi.switch(model);
    setCurrentModel(model);
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Providers</h1>
      <p className="mb-6 text-xs text-muted">Manage LLM provider API keys and select models</p>

      <div className="mb-4 rounded-sm border border-primary/30 bg-primary-dim px-4 py-2 text-xs">
        Active model: <span className="font-semibold text-primary">{currentModel}</span>
      </div>

      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {list.map((p) => (
          <div key={p.name} className="rounded-sm border border-border bg-card p-4 transition-colors hover:border-primary/30">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span>{p.emoji}</span>
                <span className="text-sm font-semibold">{p.name}</span>
              </div>
              {p.configured ? (
                <span className="flex items-center gap-1 text-[10px] text-success"><Check className="h-3 w-3" /> Ready</span>
              ) : (
                <span className="flex items-center gap-1 text-[10px] text-muted"><X className="h-3 w-3" /> No key</span>
              )}
            </div>

            <div className="mt-3 flex gap-2">
              <button onClick={() => fetchModels(p.name)}
                className="flex items-center gap-1 rounded-sm border border-border px-2 py-1 text-[10px] text-muted hover:border-primary hover:text-primary">
                <RefreshCw className="h-3 w-3" /> Models
              </button>
              <button onClick={() => { setSelected(p.name); setModelsList([]); }}
                className="flex items-center gap-1 rounded-sm border border-border px-2 py-1 text-[10px] text-muted hover:border-primary hover:text-primary">
                <Key className="h-3 w-3" /> Set Key
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Key input modal */}
      {selected && !modelsList.length && (
        <div className="mt-6 rounded-sm border border-border bg-card p-4">
          <p className="text-xs text-muted mb-2">Store API key for {selected}</p>
          <div className="flex gap-2">
            <input type="password" value={keyInput} onChange={(e) => setKeyInput(e.target.value)}
              placeholder="sk-..." className="flex-1 rounded-sm border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary" />
            <button onClick={handleStoreKey} disabled={!keyInput}
              className="rounded-sm bg-primary px-4 py-2 text-sm text-white hover:bg-primary-hover disabled:opacity-30">Save</button>
          </div>
        </div>
      )}

      {/* Models list */}
      {modelsList.length > 0 && (
        <div className="mt-6 rounded-sm border border-border bg-card p-4">
          <p className="mb-3 text-xs text-muted">{selected} models ({modelsList.length})</p>
          <div className="max-h-64 overflow-y-auto space-y-1">
            {modelsList.map((m) => (
              <button key={m} onClick={() => switchModel(m)}
                className={`block w-full rounded-sm px-3 py-1.5 text-left text-xs transition-colors ${
                  m === currentModel ? "bg-primary-dim text-primary" : "hover:bg-card-hover text-muted hover:text-foreground"
                }`}>
                {m}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
