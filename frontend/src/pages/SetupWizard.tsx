import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { providers as providersApi, models as modelsApi, getToken, system } from "@/lib/api";
import Mascot from "@/components/Mascot";
import { Check, Loader2, Database, Monitor, Bot, Cpu, Gauge, Puzzle } from "lucide-react";

/**
 * SetupWizard — mirrors CLI /setup flow:
 * 1. Storage (Local SQLite vs Cloudflare D1)
 * 2. Interface (CLI / Telegram / WhatsApp / Both)
 * 3. AI Provider + Key
 * 4. Model selection
 * 5. Limits (max RPM, max iterations)
 * 6. Composio (optional)
 *
 * If D1 is chosen + connected, loads existing config from D1 and pre-fills
 * all subsequent steps. Already-configured steps are marked with checkmarks
 * and user can press Enter/Skip to keep them.
 */

type Step = "storage" | "interface" | "provider" | "model" | "limits" | "composio";

const STEPS: { key: Step; label: string; icon: any }[] = [
  { key: "storage", label: "Storage", icon: Database },
  { key: "interface", label: "Interface", icon: Monitor },
  { key: "provider", label: "AI Provider", icon: Bot },
  { key: "model", label: "Model", icon: Cpu },
  { key: "limits", label: "Limits", icon: Gauge },
  { key: "composio", label: "Composio", icon: Puzzle },
];

interface ProviderInfo {
  name: string;
  slug: string;
  emoji: string;
  configured: boolean;
  key_name: string;
}

interface SetupConfig {
  storage_mode: string;
  interface_mode: string;
  has_provider: boolean;
  has_model: boolean;
  default_model: string;
  max_rpm: string;
  max_tool_iterations: string;
  has_composio: boolean;
  has_d1: boolean;
  configured: boolean;
  // D1 fields
  cf_account_id: string;
  cf_d1_database_id: string;
  cf_api_token: string;
}

export default function SetupWizard() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>("storage");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  // Config state (mirrors CLI config store)
  const [storageMode, setStorageMode] = useState("local");
  const [cfAccountId, setCfAccountId] = useState("");
  const [cfApiToken, setCfApiToken] = useState("");
  const [cfD1DatabaseId, setCfD1DatabaseId] = useState("");
  const [interfaceMode, setInterfaceMode] = useState("cli");
  const [selectedProvider, setSelectedProvider] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [providersList, setProvidersList] = useState<ProviderInfo[]>([]);
  const [modelsList, setModelsList] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [maxRpm, setMaxRpm] = useState("0");
  const [maxIterations, setMaxIterations] = useState("10");
  const [composioKey, setComposioKey] = useState("");

  // Track what's already configured
  const [configState, setConfigState] = useState<Partial<SetupConfig>>({});

  useEffect(() => {
    loadCurrentConfig();
  }, []);

  async function loadCurrentConfig() {
    try {
      const token = getToken();
      const [statusRes, configRes, providersRes] = await Promise.all([
        fetch("/api/setup/status", { headers: { "X-API-Token": token } }).then(r => r.json()),
        fetch("/api/system/config", { headers: { "X-API-Token": token } }).then(r => r.json()),
        fetch("/api/providers", { headers: { "X-API-Token": token } }).then(r => r.json()),
      ]);

      setConfigState(statusRes);
      setProvidersList(providersRes.providers || []);

      // Pre-fill from existing config
      if (configRes.storage_mode) setStorageMode(configRes.storage_mode);
      if (configRes.interface_mode) setInterfaceMode(configRes.interface_mode);
      if (configRes.default_model) setSelectedModel(configRes.default_model);
      if (configRes.max_rpm) setMaxRpm(String(configRes.max_rpm));
      if (configRes.max_tool_iterations) setMaxIterations(String(configRes.max_tool_iterations));

      // Find first configured provider
      const configured = (providersRes.providers || []).find((p: ProviderInfo) => p.configured);
      if (configured) setSelectedProvider(configured.name);

      // If fully configured, skip to app
      if (statusRes.configured) {
        navigate("/", { replace: true });
        return;
      }

      // Jump to first unconfigured step
      if (statusRes.has_d1 || configRes.storage_mode) {
        if (statusRes.has_provider) {
          if (statusRes.has_model) {
            setStep("limits");
          } else {
            setStep("model");
          }
        } else {
          setStep("provider");
        }
      }
    } catch {
      // Can't load config — start from beginning
    } finally {
      setLoading(false);
    }
  }

  function isStepDone(s: Step): boolean {
    switch (s) {
      case "storage": return !!(configState as any)?.has_d1 || storageMode === "local";
      case "interface": return !!interfaceMode;
      case "provider": return !!(configState as any)?.has_provider || !!selectedProvider;
      case "model": return !!(configState as any)?.has_model || !!selectedModel;
      case "limits": return true; // always has defaults
      case "composio": return true; // optional
    }
  }

  function currentStepIndex() {
    return STEPS.findIndex(s => s.key === step);
  }

  function nextStep() {
    const idx = currentStepIndex();
    if (idx < STEPS.length - 1) {
      setStep(STEPS[idx + 1].key);
      setError("");
    } else {
      finishSetup();
    }
  }

  function prevStep() {
    const idx = currentStepIndex();
    if (idx > 0) {
      setStep(STEPS[idx - 1].key);
      setError("");
    }
  }

  async function saveStorageConfig() {
    setSaving(true);
    try {
      const token = getToken();
      await fetch("/api/system/config", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Token": token },
        body: JSON.stringify({ key: "storage_mode", value: storageMode }),
      });
      // If D1, save D1 credentials
      if (storageMode === "cloudflare" && cfAccountId && cfApiToken && cfD1DatabaseId) {
        // Store D1 config via memory endpoint
        await Promise.all([
          fetch("/api/memory", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "cf_account_id", value: cfAccountId }) }),
          fetch("/api/memory", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "cf_d1_database_id", value: cfD1DatabaseId }) }),
          fetch("/api/credentials", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ name: "CF_API_TOKEN", value: cfApiToken }) }),
        ]);
      }
      nextStep();
    } catch (err: any) {
      setError(err.message || "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  async function saveInterfaceConfig() {
    setSaving(true);
    try {
      const token = getToken();
      await fetch("/api/system/config", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Token": token },
        body: JSON.stringify({ key: "interface_mode", value: interfaceMode }),
      });
      nextStep();
    } catch (err: any) {
      setError(err.message || "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  async function saveProviderKey() {
    if (!selectedProvider || !apiKey.trim()) {
      setError("Select a provider and enter an API key");
      return;
    }
    setSaving(true);
    setError("");
    try {
      await providersApi.storeKey(selectedProvider, apiKey.trim());
      // Fetch models for this provider
      const res = await providersApi.models(selectedProvider);
      setModelsList(res.models || []);
      nextStep();
    } catch (err: any) {
      setError(err.message || "Failed to save key");
    } finally {
      setSaving(false);
    }
  }

  async function saveModel() {
    if (!selectedModel) { setError("Select a model"); return; }
    setSaving(true);
    setError("");
    try {
      await modelsApi.switch(selectedModel);
      nextStep();
    } catch (err: any) {
      setError(err.message || "Failed to save model");
    } finally {
      setSaving(false);
    }
  }

  async function saveLimits() {
    setSaving(true);
    try {
      const token = getToken();
      // These may fail if keys not in allowed list, that's ok
      await fetch("/api/system/config", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "max_rpm", value: maxRpm }) }).catch(() => {});
      await fetch("/api/system/config", { method: "POST", headers: { "Content-Type": "application/json", "X-API-Token": token }, body: JSON.stringify({ key: "max_tool_iterations", value: maxIterations }) }).catch(() => {});
      nextStep();
    } finally {
      setSaving(false);
    }
  }

  async function saveComposio() {
    if (composioKey.trim()) {
      setSaving(true);
      try {
        const token = getToken();
        await fetch("/api/credentials", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-Token": token },
          body: JSON.stringify({ name: "COMPOSIO_API_KEY", value: composioKey.trim() }),
        });
      } catch {}
      setSaving(false);
    }
    finishSetup();
  }

  function finishSetup() {
    navigate("/", { replace: true });
  }

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-lg space-y-6">
        {/* Header */}
        <div className="text-center">
          <Mascot className="mx-auto mb-3" />
          <h1 className="text-lg font-bold text-primary">[+] Setup Wizard</h1>
          <p className="mt-1 text-[10px] text-muted">Configure your SynthClaw agent</p>
        </div>

        {/* Step indicator */}
        <div className="flex items-center justify-center gap-1 flex-wrap">
          {STEPS.map((s, i) => {
            const Icon = s.icon;
            const active = s.key === step;
            const done = isStepDone(s.key) && currentStepIndex() > i;
            return (
              <div key={s.key} className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-[9px] font-medium ${
                active ? "bg-primary/20 text-primary" : done ? "bg-success/15 text-success" : "text-muted"
              }`}>
                {done && !active ? <Check className="h-2.5 w-2.5" /> : <Icon className="h-2.5 w-2.5" />}
                {s.label}
              </div>
            );
          })}
        </div>

        {/* Step content */}
        <div className="rounded-sm border border-border bg-card p-5">
          {/* STEP 1: Storage */}
          {step === "storage" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">1. Storage</h2>
              <p className="text-[10px] text-muted">Where should SynthClaw store data?</p>
              <div className="space-y-2">
                <label className={`flex items-center gap-3 rounded-sm border p-3 cursor-pointer transition-colors ${storageMode === "local" ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                  <input type="radio" name="storage" value="local" checked={storageMode === "local"} onChange={() => setStorageMode("local")} className="accent-primary" />
                  <div>
                    <p className="text-xs font-medium">Local SQLite</p>
                    <p className="text-[10px] text-muted">Default. Data stays on this server.</p>
                  </div>
                </label>
                <label className={`flex items-center gap-3 rounded-sm border p-3 cursor-pointer transition-colors ${storageMode === "cloudflare" ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                  <input type="radio" name="storage" value="cloudflare" checked={storageMode === "cloudflare"} onChange={() => setStorageMode("cloudflare")} className="accent-primary" />
                  <div>
                    <p className="text-xs font-medium">Cloudflare D1 + R2</p>
                    <p className="text-[10px] text-muted">Cloud sync. Config loaded from D1 on setup.</p>
                  </div>
                </label>
              </div>

              {storageMode === "cloudflare" && (
                <div className="space-y-2 pt-2 border-t border-border">
                  <input value={cfAccountId} onChange={e => setCfAccountId(e.target.value)} placeholder="Cloudflare Account ID"
                    className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                  <input type="password" value={cfApiToken} onChange={e => setCfApiToken(e.target.value)} placeholder="Cloudflare API Token"
                    className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                  <input value={cfD1DatabaseId} onChange={e => setCfD1DatabaseId(e.target.value)} placeholder="D1 Database ID"
                    className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                </div>
              )}

              {error && <p className="text-[10px] text-danger">{error}</p>}
              <button onClick={saveStorageConfig} disabled={saving || (storageMode === "cloudflare" && (!cfAccountId || !cfApiToken || !cfD1DatabaseId))}
                className="w-full rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                {saving ? "Saving..." : "Continue"}
              </button>
            </div>
          )}

          {/* STEP 2: Interface */}
          {step === "interface" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">2. Interface</h2>
              <p className="text-[10px] text-muted">How will you interact with SynthClaw?</p>
              <div className="space-y-2">
                {[
                  { value: "cli", label: "CLI only", desc: "No messaging platform" },
                  { value: "telegram", label: "Telegram", desc: "Telegram bot interface" },
                  { value: "whatsapp", label: "WhatsApp", desc: "WhatsApp Business API" },
                  { value: "both", label: "Both", desc: "Telegram + WhatsApp" },
                ].map(opt => (
                  <label key={opt.value} className={`flex items-center gap-3 rounded-sm border p-3 cursor-pointer transition-colors ${interfaceMode === opt.value ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                    <input type="radio" name="interface" value={opt.value} checked={interfaceMode === opt.value} onChange={() => setInterfaceMode(opt.value)} className="accent-primary" />
                    <div>
                      <p className="text-xs font-medium">{opt.label}</p>
                      <p className="text-[10px] text-muted">{opt.desc}</p>
                    </div>
                  </label>
                ))}
              </div>
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={prevStep} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveInterfaceConfig} disabled={saving}
                  className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                  {saving ? "Saving..." : "Continue"}
                </button>
              </div>
            </div>
          )}

          {/* STEP 3: AI Provider */}
          {step === "provider" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">3. AI Provider</h2>
              <p className="text-[10px] text-muted">Select provider and enter API key</p>
              <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                {providersList.map(p => (
                  <button key={p.name} onClick={() => setSelectedProvider(p.name)}
                    className={`flex items-center gap-2 rounded-sm border px-3 py-2 text-left text-xs transition-colors ${selectedProvider === p.name ? "border-primary bg-primary/5" : "border-border hover:border-muted"}`}>
                    <span>{p.emoji}</span>
                    <span className="truncate font-medium">{p.name}</span>
                    {p.configured && <Check className="h-3 w-3 text-success ml-auto" />}
                  </button>
                ))}
              </div>
              {selectedProvider && (
                <div className="pt-2">
                  <label className="text-[10px] text-muted block mb-1">{selectedProvider} API Key</label>
                  <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
                    placeholder={providersList.find(p => p.name === selectedProvider)?.configured ? "Key already stored (enter to replace)" : "sk-..."}
                    className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary"
                    onKeyDown={e => e.key === "Enter" && saveProviderKey()} />
                </div>
              )}
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={prevStep} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={() => {
                  // If provider already configured and no new key entered, skip
                  const prov = providersList.find(p => p.name === selectedProvider);
                  if (prov?.configured && !apiKey.trim()) {
                    providersApi.models(selectedProvider).then(r => { setModelsList(r.models || []); nextStep(); }).catch(() => nextStep());
                  } else {
                    saveProviderKey();
                  }
                }} disabled={saving || !selectedProvider}
                  className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                  {saving ? "Saving..." : (providersList.find(p => p.name === selectedProvider)?.configured && !apiKey.trim()) ? "Keep & Continue" : "Save & Continue"}
                </button>
              </div>
            </div>
          )}

          {/* STEP 4: Model */}
          {step === "model" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">4. Default Model</h2>
              <p className="text-[10px] text-muted">
                Select a model from {selectedProvider || "your provider"}
                {selectedModel && <span className="text-success ml-2">(current: {selectedModel})</span>}
              </p>
              <div className="max-h-52 overflow-y-auto space-y-1">
                {modelsList.map(m => (
                  <button key={m} onClick={() => setSelectedModel(m)}
                    className={`w-full rounded-sm border px-3 py-1.5 text-left text-xs transition-colors ${selectedModel === m ? "border-primary bg-primary/5 text-primary" : "border-border text-foreground hover:border-muted"}`}>
                    {m}
                  </button>
                ))}
                {modelsList.length === 0 && (
                  <p className="text-center text-[10px] text-muted py-4">No models found. You can type one manually below.</p>
                )}
              </div>
              <input value={selectedModel} onChange={e => setSelectedModel(e.target.value)} placeholder="Or type model ID manually..."
                className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={prevStep} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveModel} disabled={saving || !selectedModel}
                  className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                  {saving ? "Saving..." : "Continue"}
                </button>
              </div>
            </div>
          )}

          {/* STEP 5: Limits */}
          {step === "limits" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">5. Limits</h2>
              <p className="text-[10px] text-muted">Rate limiting and tool iteration caps</p>
              <div className="space-y-3">
                <div>
                  <label className="text-[10px] text-muted block mb-1">Max requests/minute (0 = unlimited)</label>
                  <input value={maxRpm} onChange={e => setMaxRpm(e.target.value)} placeholder="0"
                    className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                </div>
                <div>
                  <label className="text-[10px] text-muted block mb-1">Max tool iterations per message</label>
                  <input value={maxIterations} onChange={e => setMaxIterations(e.target.value)} placeholder="10"
                    className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
                </div>
              </div>
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={prevStep} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveLimits} disabled={saving}
                  className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                  {saving ? "Saving..." : "Continue"}
                </button>
              </div>
            </div>
          )}

          {/* STEP 6: Composio */}
          {step === "composio" && (
            <div className="space-y-4">
              <h2 className="text-sm font-bold">6. Composio (optional)</h2>
              <p className="text-[10px] text-muted">200+ app integrations (GitHub, Slack, Gmail, etc.)</p>
              <input value={composioKey} onChange={e => setComposioKey(e.target.value)} placeholder="Composio API Key (Enter to skip)"
                className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary"
                onKeyDown={e => e.key === "Enter" && saveComposio()} />
              <p className="text-[10px] text-muted">Get your key at composio.dev — skip if you don't need app integrations.</p>
              {error && <p className="text-[10px] text-danger">{error}</p>}
              <div className="flex gap-2">
                <button onClick={prevStep} className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground">Back</button>
                <button onClick={saveComposio} disabled={saving}
                  className="flex-1 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
                  {saving ? "Saving..." : composioKey.trim() ? "Save & Finish" : "Skip & Finish"}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Skip all */}
        <button onClick={finishSetup} className="block mx-auto text-[10px] text-muted hover:text-foreground">
          Skip setup for now
        </button>
      </div>
    </div>
  );
}
