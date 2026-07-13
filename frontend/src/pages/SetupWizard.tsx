import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { providers as providersApi, models as modelsApi, getToken } from "@/lib/api";
import Mascot from "@/components/Mascot";
import { Check, Loader2, Database, Monitor, Bot, Cpu, Gauge, Puzzle } from "lucide-react";

type Step = "storage" | "checking" | "interface" | "provider" | "model" | "limits" | "composio";

const VISIBLE_STEPS: { key: Step; label: string; icon: any }[] = [
  { key: "storage", label: "Storage", icon: Database },
  { key: "interface", label: "Interface", icon: Monitor },
  { key: "provider", label: "Provider", icon: Bot },
  { key: "model", label: "Model", icon: Cpu },
  { key: "limits", label: "Limits", icon: Gauge },
  { key: "composio", label: "Composio", icon: Puzzle },
];

interface ProviderInfo {
  name: string; slug: string; emoji: string; configured: boolean; key_name: string;
}

interface DbConfig {
  interface_mode?: string;
  has_provider?: boolean;
  provider_name?: string;
  has_model?: boolean;
  default_model?: string;
  storage_mode?: string;
  has_d1?: boolean;
  has_composio?: boolean;
  max_rpm?: string;
  max_tool_iterations?: string;
}


export default function SetupWizard() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>("storage");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  // Storage
  const [storageMode, setStorageMode] = useState("local");
  const [cfAccountId, setCfAccountId] = useState("");
  const [cfApiToken, setCfApiToken] = useState("");
  const [cfD1DatabaseId, setCfD1DatabaseId] = useState("");

  // Loaded from DB after storage step
  const [dbConfig, setDbConfig] = useState<DbConfig>({});
  const [missingSteps, setMissingSteps] = useState<Step[]>([]);

  // Interface
  const [interfaceMode, setInterfaceMode] = useState("cli");
  // Provider
  const [selectedProvider, setSelectedProvider] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [providersList, setProvidersList] = useState<ProviderInfo[]>([]);
  // Model
  const [modelsList, setModelsList] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  // Limits
  const [maxRpm, setMaxRpm] = useState("0");
  const [maxIterations, setMaxIterations] = useState("10");
  // Composio
  const [composioKey, setComposioKey] = useState("");


  // After storage is selected, check the DB and determine which steps are missing
  async function checkDatabaseAndProceed() {
    setStep("checking");
    setLoading(true);
    setError("");
    try {
      const token = getToken();
      // Save storage config first
      await fetch("/api/system/config", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Token": token },
        body: JSON.stringify({ key: "storage_mode", value: storageMode }),
      });
      if (storageMode === "cloudflare" && cfAccountId && cfApiToken && cfD1DatabaseId) {
        await Promise.all([
          fetch("/api/memory", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "cf_account_id", value: cfAccountId }) }),
          fetch("/api/memory", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "cf_d1_database_id", value: cfD1DatabaseId }) }),
          fetch("/api/credentials", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ name: "CF_API_TOKEN", value: cfApiToken }) }),
        ]);
      }

      // Now query the database to see what's already configured
      const [statusRes, configRes, providersRes] = await Promise.all([
        fetch("/api/setup/status", { headers: { "X-API-Token": token } }).then(r => r.json()).catch(() => ({})),
        fetch("/api/system/config", { headers: { "X-API-Token": token } }).then(r => r.json()).catch(() => ({})),
        fetch("/api/providers", { headers: { "X-API-Token": token } }).then(r => r.json()).catch(() => ({ providers: [] })),
      ]);

      const cfg: DbConfig = { ...statusRes, ...configRes };
      setDbConfig(cfg);
      setProvidersList(providersRes.providers || []);

      // Pre-fill from DB
      if (cfg.interface_mode) setInterfaceMode(cfg.interface_mode);
      if (cfg.default_model) setSelectedModel(cfg.default_model);
      if (cfg.max_rpm) setMaxRpm(String(cfg.max_rpm));
      if (cfg.max_tool_iterations) setMaxIterations(String(cfg.max_tool_iterations));
      const configuredProv = (providersRes.providers || []).find((p: ProviderInfo) => p.configured);
      if (configuredProv) setSelectedProvider(configuredProv.name);


      // Determine which steps are MISSING (not yet configured)
      const missing: Step[] = [];
      if (!cfg.interface_mode) missing.push("interface");
      if (!cfg.has_provider && !configuredProv) missing.push("provider");
      if (!cfg.has_model && !cfg.default_model) missing.push("model");
      // Limits always have defaults, but include if max_rpm is not set
      if (!cfg.max_rpm && cfg.max_rpm !== "0") missing.push("limits");
      if (!cfg.has_composio) missing.push("composio");

      setMissingSteps(missing);

      // If nothing is missing, we're done
      if (missing.length === 0) {
        navigate("/", { replace: true });
        return;
      }

      // Jump to first missing step
      setStep(missing[0]);
    } catch (err: any) {
      setError("Failed to check database: " + (err.message || ""));
      setStep("storage");
    } finally {
      setLoading(false);
    }
  }

  function goToNextMissing() {
    const currentIdx = missingSteps.indexOf(step as any);
    if (currentIdx < missingSteps.length - 1) {
      setStep(missingSteps[currentIdx + 1]);
      setError("");
    } else {
      navigate("/", { replace: true });
    }
  }

  function goToPrevMissing() {
    const currentIdx = missingSteps.indexOf(step as any);
    if (currentIdx > 0) {
      setStep(missingSteps[currentIdx - 1]);
      setError("");
    } else {
      setStep("storage");
      setError("");
    }
  }


  async function saveInterface() {
    setSaving(true);
    try {
      const token = getToken();
      await fetch("/api/system/config", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "interface_mode", value: interfaceMode }) });
      goToNextMissing();
    } catch (err: any) { setError(err.message || "Failed"); }
    finally { setSaving(false); }
  }

  async function saveProvider() {
    if (!selectedProvider || !apiKey.trim()) { setError("Select provider and enter key"); return; }
    setSaving(true); setError("");
    try {
      await providersApi.storeKey(selectedProvider, apiKey.trim());
      const res = await providersApi.models(selectedProvider);
      setModelsList(res.models || []);
      goToNextMissing();
    } catch (err: any) { setError(err.message || "Failed"); }
    finally { setSaving(false); }
  }

  async function saveModel() {
    if (!selectedModel) { setError("Select a model"); return; }
    setSaving(true); setError("");
    try {
      await modelsApi.switch(selectedModel);
      goToNextMissing();
    } catch (err: any) { setError(err.message || "Failed"); }
    finally { setSaving(false); }
  }

  async function saveLimits() {
    setSaving(true);
    const token = getToken();
    await fetch("/api/system/config", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "max_rpm", value: maxRpm }) }).catch(() => {});
    await fetch("/api/system/config", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "max_tool_iterations", value: maxIterations }) }).catch(() => {});
    setSaving(false);
    goToNextMissing();
  }

  async function saveComposio() {
    if (composioKey.trim()) {
      setSaving(true);
      const token = getToken();
      await fetch("/api/credentials", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ name: "COMPOSIO_API_KEY", value: composioKey.trim() }) }).catch(() => {});
      setSaving(false);
    }
    navigate("/", { replace: true });
  }


  // --- RENDER ---
  if (step === "checking") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center">
          <Loader2 className="h-5 w-5 animate-spin text-primary mx-auto" />
          <p className="mt-3 text-xs text-muted">Checking database for existing config...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-lg space-y-6">
        <div className="text-center">
          <Mascot className="mx-auto mb-3" />
          <h1 className="text-lg font-bold text-primary">[+] Setup Wizard</h1>
          <p className="mt-1 text-[10px] text-muted">
            {missingSteps.length > 0
              ? `${missingSteps.length} step${missingSteps.length > 1 ? "s" : ""} remaining`
              : "Configure your SynthClaw agent"}
          </p>
        </div>

        {/* Step indicators */}
        <div className="flex items-center justify-center gap-1 flex-wrap">
          {VISIBLE_STEPS.map((s) => {
            const Icon = s.icon;
            const active = s.key === step;
            const isMissing = missingSteps.includes(s.key);
            const done = !isMissing && step !== "storage";
            return (
              <div key={s.key} className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium ${
                active ? "bg-primary/20 text-primary" : done ? "bg-success/15 text-success" : isMissing ? "text-muted" : "text-muted/50"
              }`}>
                {done ? <Check className="h-2.5 w-2.5" /> : <Icon className="h-2.5 w-2.5" />}
                {s.label}
              </div>
            );
          })}
        </div>

        <div className="rounded-sm border border-border bg-card p-5">


          {/* STEP: Storage */}
          {step === "storage" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">1. Storage</h2>
              <p className="text-[10px] text-muted">Where is your data stored? After selecting, we'll check what's already configured.</p>
              <div className="space-y-2">
                <label className={`flex items-center gap-3 rounded-sm border p-3 cursor-pointer ${storageMode === "local" ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                  <input type="radio" name="storage" value="local" checked={storageMode === "local"} onChange={() => setStorageMode("local")} className="accent-primary" />
                  <div><p className="text-xs font-medium">Local SQLite</p><p className="text-[10px] text-muted">Data on this server.</p></div>
                </label>
                <label className={`flex items-center gap-3 rounded-sm border p-3 cursor-pointer ${storageMode === "cloudflare" ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                  <input type="radio" name="storage" value="cloudflare" checked={storageMode === "cloudflare"} onChange={() => setStorageMode("cloudflare")} className="accent-primary" />
                  <div><p className="text-xs font-medium">Cloudflare D1 + R2</p><p className="text-[10px] text-muted">Cloud sync. Existing config loaded from D1.</p></div>
                </label>
              </div>
              {storageMode === "cloudflare" && (
                <div className="space-y-2 pt-2 border-t border-border">
                  <input value={cfAccountId} onChange={e => setCfAccountId(e.target.value)} placeholder="Cloudflare Account ID" className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                  <input type="password" value={cfApiToken} onChange={e => setCfApiToken(e.target.value)} placeholder="Cloudflare API Token" className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                  <input value={cfD1DatabaseId} onChange={e => setCfD1DatabaseId(e.target.value)} placeholder="D1 Database ID" className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                </div>
              )}
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <button onClick={checkDatabaseAndProceed} disabled={loading || (storageMode === "cloudflare" && (!cfAccountId || !cfApiToken || !cfD1DatabaseId))}
                className="w-full rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                {loading ? "Checking..." : "Check Database & Continue"}
              </button>
            </div>
          )}


          {/* STEP: Interface */}
          {step === "interface" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">Interface</h2>
              <p className="text-[10px] text-muted">How will you interact with SynthClaw?</p>
              <div className="space-y-2">
                {[
                  { value: "cli", label: "CLI only", desc: "Terminal + web frontend" },
                  { value: "frontend", label: "Frontend only", desc: "Web UI only, no messaging bots" },
                  { value: "telegram", label: "Telegram", desc: "Telegram bot interface" },
                  { value: "whatsapp", label: "WhatsApp", desc: "WhatsApp Business API" },
                  { value: "both", label: "Telegram + WhatsApp", desc: "Both messaging platforms" },
                  { value: "all", label: "All", desc: "CLI + Frontend + Telegram + WhatsApp" },
                ].map(opt => (
                  <label key={opt.value} className={`flex items-center gap-3 rounded-sm border p-3 cursor-pointer ${interfaceMode === opt.value ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                    <input type="radio" name="interface" value={opt.value} checked={interfaceMode === opt.value} onChange={() => setInterfaceMode(opt.value)} className="accent-primary" />
                    <div><p className="text-xs font-medium">{opt.label}</p><p className="text-[10px] text-muted">{opt.desc}</p></div>
                  </label>
                ))}
              </div>
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={goToPrevMissing} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveInterface} disabled={saving} className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">{saving ? "..." : "Continue"}</button>
              </div>
            </div>
          )}


          {/* STEP: Provider */}
          {step === "provider" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">AI Provider</h2>
              <p className="text-[10px] text-muted">Select provider and enter API key</p>
              <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                {providersList.map(p => (
                  <button key={p.name} onClick={() => setSelectedProvider(p.name)}
                    className={`flex items-center gap-2 rounded-sm border px-3 py-2 text-left text-xs ${selectedProvider === p.name ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                    <span>{p.emoji}</span><span className="truncate font-medium">{p.name}</span>
                    {p.configured && <Check className="h-3 w-3 text-success ml-auto" />}
                  </button>
                ))}
              </div>
              {selectedProvider && (
                <div className="pt-2">
                  <label className="text-[10px] text-muted block mb-1">{selectedProvider} API Key</label>
                  <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
                    placeholder={providersList.find(p => p.name === selectedProvider)?.configured ? "Already stored (enter to replace)" : "sk-..."}
                    className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary"
                    onKeyDown={e => e.key === "Enter" && saveProvider()} />
                </div>
              )}
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={goToPrevMissing} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={() => {
                  const prov = providersList.find(p => p.name === selectedProvider);
                  if (prov?.configured && !apiKey.trim()) {
                    providersApi.models(selectedProvider).then(r => { setModelsList(r.models || []); goToNextMissing(); }).catch(() => goToNextMissing());
                  } else { saveProvider(); }
                }} disabled={saving || !selectedProvider} className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                  {saving ? "..." : providersList.find(p => p.name === selectedProvider)?.configured && !apiKey.trim() ? "Keep & Continue" : "Save & Continue"}
                </button>
              </div>
            </div>
          )}


          {/* STEP: Model */}
          {step === "model" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">Default Model</h2>
              <p className="text-[10px] text-muted">
                {selectedModel && <span className="text-success">Current: {selectedModel} — </span>}
                Select or type a model ID
              </p>
              <div className="max-h-44 overflow-y-auto space-y-1">
                {modelsList.map(m => (
                  <button key={m} onClick={() => setSelectedModel(m)}
                    className={`w-full rounded-sm border px-3 py-1.5 text-left text-xs ${selectedModel === m ? "border-primary bg-primary/5 text-primary" : "border-border hover:border-muted"}`}>{m}</button>
                ))}
                {modelsList.length === 0 && <p className="text-center text-[10px] text-muted py-3">No models fetched. Type manually below.</p>}
              </div>
              <input value={selectedModel} onChange={e => setSelectedModel(e.target.value)} placeholder="Model ID..."
                className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={goToPrevMissing} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveModel} disabled={saving || !selectedModel} className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">{saving ? "..." : "Continue"}</button>
              </div>
            </div>
          )}

          {/* STEP: Limits */}
          {step === "limits" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">Limits</h2>
              <p className="text-[10px] text-muted">Rate limiting and tool caps</p>
              <div className="space-y-3">
                <div><label className="text-[10px] text-muted block mb-1">Max requests/min (0 = unlimited)</label>
                  <input value={maxRpm} onChange={e => setMaxRpm(e.target.value)} className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" /></div>
                <div><label className="text-[10px] text-muted block mb-1">Max tool iterations/message</label>
                  <input value={maxIterations} onChange={e => setMaxIterations(e.target.value)} className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" /></div>
              </div>
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={goToPrevMissing} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveLimits} disabled={saving} className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">{saving ? "..." : "Continue"}</button>
              </div>
            </div>
          )}

          {/* STEP: Composio */}
          {step === "composio" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">Composio (optional)</h2>
              <p className="text-[10px] text-muted">200+ app integrations — GitHub, Slack, Gmail, etc.</p>
              <input value={composioKey} onChange={e => setComposioKey(e.target.value)} placeholder="Composio API Key (Enter to skip)"
                className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary"
                onKeyDown={e => e.key === "Enter" && saveComposio()} />
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={goToPrevMissing} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveComposio} disabled={saving} className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">{saving ? "..." : composioKey.trim() ? "Save & Finish" : "Skip & Finish"}</button>
              </div>
            </div>
          )}
        </div>

        <button onClick={() => navigate("/", { replace: true })} className="block mx-auto text-[10px] text-muted hover:text-foreground">Skip setup</button>
      </div>
    </div>
  );
}
