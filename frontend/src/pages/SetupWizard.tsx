import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { providers as providersApi, models as modelsApi, getToken } from "@/lib/api";
import Mascot from "@/components/Mascot";
import { ChevronRight, Check, Loader2 } from "lucide-react";

type Step = "provider" | "key" | "model";

interface ProviderInfo {
  name: string;
  slug: string;
  emoji: string;
  configured: boolean;
  key_name: string;
}

interface SetupState {
  has_provider: boolean;
  has_model: boolean;
  configured: boolean;
}

export default function SetupWizard() {
  const navigate = useNavigate();
  const [step, setStep] = useState<Step>("provider");
  const [providersList, setProvidersList] = useState<ProviderInfo[]>([]);
  const [selectedProvider, setSelectedProvider] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [modelsList, setModelsList] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(true);
  const [setupState, setSetupState] = useState<SetupState>({ has_provider: false, has_model: false, configured: false });

  // Check setup status on mount — skip wizard if fully configured, or jump to the right step
  useEffect(() => {
    fetch("/api/setup/status", {
      headers: { "X-API-Token": getToken() },
    })
      .then((r) => r.json())
      .then((data: SetupState) => {
        setSetupState(data);
        if (data.configured) {
          // Fully configured — skip wizard entirely
          navigate("/", { replace: true });
        } else if (data.has_provider && !data.has_model) {
          // Provider done, jump to model step
          setStep("model");
          setChecking(false);
        } else {
          // Need provider setup
          setChecking(false);
        }
      })
      .catch(() => setChecking(false));
  }, []);

  // Load providers list
  useEffect(() => {
    if (!checking) {
      providersApi.list().then((r) => {
        setProvidersList(r.providers || []);
        // If a provider is already configured, pre-select it
        const configured = (r.providers || []).find((p: ProviderInfo) => p.configured);
        if (configured) {
          setSelectedProvider(configured.name);
        }
      }).catch(() => {});
    }
  }, [checking]);

  // If we jumped to model step, load models for configured provider
  useEffect(() => {
    if (step === "model" && setupState.has_provider && !modelsList.length) {
      // Find the configured provider and fetch its models
      const configured = providersList.find((p) => p.configured);
      if (configured) {
        setSelectedProvider(configured.name);
        providersApi.models(configured.name).then((r) => setModelsList(r.models || [])).catch(() => {});
      }
    }
  }, [step, providersList, setupState.has_provider]);

  async function handleKeySubmit() {
    if (!apiKey.trim()) { setError("API key is required"); return; }
    setLoading(true);
    setError("");
    try {
      await providersApi.storeKey(selectedProvider, apiKey.trim());
      // Fetch models for provider
      const res = await providersApi.models(selectedProvider);
      setModelsList(res.models || []);
      setStep("model");
    } catch (err: any) {
      setError(err.message || "Failed to save key");
    } finally {
      setLoading(false);
    }
  }

  async function handleModelSelect(model: string) {
    setSelectedModel(model);
    setLoading(true);
    setError("");
    try {
      await modelsApi.switch(model);
      navigate("/", { replace: true });
    } catch (err: any) {
      setError(err.message || "Failed to set model");
      setLoading(false);
    }
  }

  function handleSkip() {
    navigate("/", { replace: true });
  }

  if (checking) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Header */}
        <div className="text-center">
          <Mascot className="mx-auto mb-4" />
          <h1 className="text-xl font-bold text-primary">[+] Setup Wizard</h1>
          <p className="mt-1 text-xs text-muted">Configure your AI provider to get started</p>
        </div>

        {/* Step indicator */}
        <div className="flex items-center justify-center gap-2 text-[10px]">
          <StepBadge label="Provider" active={step === "provider"} done={setupState.has_provider || step === "key" || step === "model"} />
          <ChevronRight className="h-3 w-3 text-muted" />
          <StepBadge label="API Key" active={step === "key"} done={setupState.has_provider || step === "model"} />
          <ChevronRight className="h-3 w-3 text-muted" />
          <StepBadge label="Model" active={step === "model"} done={setupState.has_model} />
        </div>

        {/* Step: Provider selection */}
        {step === "provider" && (
          <div className="space-y-3">
            <p className="text-xs text-muted text-center">Choose your LLM provider</p>
            <div className="grid grid-cols-2 gap-2 max-h-80 overflow-y-auto">
              {providersList.map((p) => (
                <button
                  key={p.name}
                  onClick={() => {
                    setSelectedProvider(p.name);
                    if (p.configured) {
                      // Already has a key — jump straight to model
                      setStep("model");
                      providersApi.models(p.name).then((r) => setModelsList(r.models || [])).catch(() => {});
                    } else {
                      setStep("key");
                    }
                  }}
                  className="flex items-center gap-2 rounded-sm border border-border bg-card px-3 py-2.5 text-left text-xs hover:border-primary transition-colors"
                >
                  <span className="text-base">{p.emoji}</span>
                  <span className="text-foreground font-medium truncate">{p.name}</span>
                  {p.configured && <Check className="h-3 w-3 text-success ml-auto flex-shrink-0" />}
                </button>
              ))}
            </div>
            {providersList.length === 0 && (
              <p className="text-center text-xs text-muted py-4">Loading providers...</p>
            )}
            {/* Skip link */}
            <button onClick={handleSkip} className="block mx-auto text-[10px] text-muted hover:text-foreground">
              Skip setup for now
            </button>
          </div>
        )}

        {/* Step: API Key */}
        {step === "key" && (
          <div className="space-y-4">
            <p className="text-xs text-muted text-center">
              Enter your <span className="text-foreground font-medium">{selectedProvider}</span> API key
            </p>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-... or your provider key"
              className="w-full rounded-sm border border-border bg-card px-3 py-2.5 text-sm text-foreground placeholder-muted/50 outline-none focus:border-primary"
              autoFocus
              onKeyDown={(e) => e.key === "Enter" && handleKeySubmit()}
            />
            {error && <p className="text-xs text-danger">{error}</p>}
            <div className="flex gap-2">
              <button
                onClick={() => { setStep("provider"); setError(""); setApiKey(""); }}
                className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground"
              >
                Back
              </button>
              <button
                onClick={handleKeySubmit}
                disabled={loading || !apiKey.trim()}
                className="flex-1 rounded-sm bg-primary px-4 py-2 text-sm font-semibold text-white hover:bg-primary-hover disabled:opacity-50"
              >
                {loading ? "Saving..." : "Save & Continue"}
              </button>
            </div>
          </div>
        )}

        {/* Step: Model selection */}
        {step === "model" && (
          <div className="space-y-4">
            <p className="text-xs text-muted text-center">
              Select a model from <span className="text-foreground font-medium">{selectedProvider}</span>
            </p>
            {error && <p className="text-xs text-danger">{error}</p>}
            <div className="max-h-64 overflow-y-auto space-y-1">
              {modelsList.map((m) => (
                <button
                  key={m}
                  onClick={() => handleModelSelect(m)}
                  disabled={loading}
                  className="w-full rounded-sm border border-border bg-card px-3 py-2 text-left text-xs text-foreground hover:border-primary transition-colors disabled:opacity-50"
                >
                  {m}
                </button>
              ))}
              {modelsList.length === 0 && (
                <p className="text-center text-xs text-muted py-4">No models found. Check your API key.</p>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => { setStep(setupState.has_provider ? "provider" : "key"); setError(""); }}
                className="rounded-sm border border-border px-4 py-2 text-xs text-muted hover:text-foreground"
              >
                Back
              </button>
              <button onClick={handleSkip} className="ml-auto text-[10px] text-muted hover:text-foreground">
                Skip
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StepBadge({ label, active, done }: { label: string; active: boolean; done: boolean }) {
  return (
    <span
      className={`rounded-full px-2.5 py-0.5 text-[10px] font-medium ${
        active
          ? "bg-primary/20 text-primary"
          : done
          ? "bg-success/20 text-success"
          : "bg-card text-muted"
      }`}
    >
      {done && !active ? <Check className="inline h-2.5 w-2.5 mr-0.5" /> : null}
      {label}
    </span>
  );
}
