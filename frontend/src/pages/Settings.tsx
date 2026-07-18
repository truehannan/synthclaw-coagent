import { useEffect, useState } from "react";
import { system, auth, credentials, memory, models, providers as providersApi, clearToken, getToken } from "@/lib/api";
import { KeyRound, Check, RotateCcw, AlertTriangle, Wand2, Save, Edit3 } from "lucide-react";

export default function Settings() {
  const [cfg, setCfg] = useState<any>(null);
  const [logs, setLogs] = useState<string[]>([]);

  // Change password state
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [pwError, setPwError] = useState("");
  const [pwSuccess, setPwSuccess] = useState(false);
  const [pwLoading, setPwLoading] = useState(false);

  // Reset state
  const [resetConfirm, setResetConfirm] = useState(false);
  const [resetting, setResetting] = useState(false);

  // Editable setup config
  const [setupValues, setSetupValues] = useState<Record<string, string>>({});
  const [editingField, setEditingField] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [savingField, setSavingField] = useState("");
  const [saveSuccess, setSaveSuccess] = useState("");

  useEffect(() => {
    system.config().then(setCfg).catch(() => {});
    system.logs(30).then((r) => setLogs(r.logs || [])).catch(() => {});
    loadSetupValues();
  }, []);

  async function loadSetupValues() {
    try {
      const token = getToken();
      const [configRes, statusRes, credsRes] = await Promise.all([
        system.config(),
        fetch("/api/setup/status", { headers: { "X-API-Token": token } }).then(r => r.json()).catch(() => ({})),
        credentials.list(),
      ]);
      const values: Record<string, string> = {};
      values["Interface Mode"] = configRes?.interface_mode || "cli";
      values["Storage Mode"] = configRes?.storage_mode || "local";
      values["Default Model"] = configRes?.default_model || "";
      values["Max RPM"] = String(configRes?.max_rpm ?? "0");
      values["Max Tool Iterations"] = String(configRes?.max_tool_iterations ?? "10");
      values["Provider"] = statusRes?.provider_name || "(not set)";
      values["Composio"] = statusRes?.has_composio ? "Configured" : "Not configured";
      values["D1 Storage"] = statusRes?.has_d1 ? "Connected" : "Not connected";
      setSetupValues(values);
    } catch {}
  }

  async function saveSetupField(label: string, value: string) {
    setSavingField(label);
    const token = getToken();
    try {
      const keyMap: Record<string, string> = {
        "Interface Mode": "interface_mode",
        "Storage Mode": "storage_mode",
        "Default Model": "default_model",
        "Max RPM": "max_rpm",
        "Max Tool Iterations": "max_tool_iterations",
      };
      const configKey = keyMap[label];
      if (configKey) {
        if (configKey === "default_model") {
          await models.switch(value);
        } else {
          await fetch("/api/system/config", {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-API-Token": token },
            body: JSON.stringify({ key: configKey, value }),
          });
        }
        setSetupValues(prev => ({ ...prev, [label]: value }));
        setSaveSuccess(label);
        setTimeout(() => setSaveSuccess(""), 2000);
      }
    } catch {}
    setSavingField("");
    setEditingField(null);
  }

  async function handleChangePassword(e: React.FormEvent) {
    e.preventDefault();
    setPwError("");
    setPwSuccess(false);
    if (newPw !== confirmPw) { setPwError("Passwords don't match"); return; }
    if (newPw.length < 4) { setPwError("Minimum 4 characters"); return; }
    setPwLoading(true);
    try {
      await auth.changePassword(currentPw, newPw);
      setPwSuccess(true);
      setCurrentPw("");
      setNewPw("");
      setConfirmPw("");
      setTimeout(() => setPwSuccess(false), 3000);
    } catch (err: any) {
      setPwError(err.message || "Failed to change password");
    } finally {
      setPwLoading(false);
    }
  }

  async function handleReset() {
    setResetting(true);
    try {
      // Delete the database and clear auth
      await system.run("rm -f /opt/agent/agent.db /opt/agent/.api_token /opt/agent/.fernet_key");
      clearToken();
      window.location.href = "/signup";
    } catch {
      // Even if command fails, redirect to signup
      clearToken();
      window.location.href = "/signup";
    }
  }

  function handleRerunSetup() {
    window.location.href = "/setup";
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Settings</h1>
      <p className="mb-6 text-xs text-muted">Account, configuration, and system logs</p>

      {/* Account — Change Password */}
      <div className="mb-6 rounded-sm border border-border bg-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <KeyRound className="h-3.5 w-3.5 text-muted" />
          <p className="text-xs font-semibold text-muted">Account</p>
        </div>
        <form onSubmit={handleChangePassword} className="space-y-3 max-w-sm">
          <div>
            <label className="mb-1 block text-[10px] text-muted">Current Password</label>
            <input
              type="password"
              value={currentPw}
              onChange={(e) => setCurrentPw(e.target.value)}
              placeholder="Current password..."
              className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs text-foreground placeholder-muted/50 outline-none focus:border-primary"
            />
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-muted">New Password</label>
            <input
              type="password"
              value={newPw}
              onChange={(e) => setNewPw(e.target.value)}
              placeholder="New password..."
              className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs text-foreground placeholder-muted/50 outline-none focus:border-primary"
            />
          </div>
          <div>
            <label className="mb-1 block text-[10px] text-muted">Confirm New Password</label>
            <input
              type="password"
              value={confirmPw}
              onChange={(e) => setConfirmPw(e.target.value)}
              placeholder="Confirm..."
              className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs text-foreground placeholder-muted/50 outline-none focus:border-primary"
            />
          </div>
          {pwError && <p className="text-[10px] text-danger">{pwError}</p>}
          {pwSuccess && (
            <p className="flex items-center gap-1 text-[10px] text-success">
              <Check className="h-3 w-3" /> Password changed successfully
            </p>
          )}
          <button
            type="submit"
            disabled={pwLoading || !currentPw || !newPw}
            className="rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50"
          >
            {pwLoading ? "Changing..." : "Change Password"}
          </button>
        </form>
      </div>

      {/* Re-run Setup / Reset */}
      <div className="mb-6 rounded-sm border border-border bg-card p-4">
        <p className="text-xs font-semibold text-muted mb-3">Setup & Reset</p>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={handleRerunSetup}
            className="flex items-center gap-2 rounded-sm border border-border px-4 py-2 text-xs text-muted hover:border-primary hover:text-primary"
          >
            <Wand2 className="h-3.5 w-3.5" /> Re-run Setup Wizard
          </button>

          {!resetConfirm ? (
            <button
              onClick={() => setResetConfirm(true)}
              className="flex items-center gap-2 rounded-sm border border-border px-4 py-2 text-xs text-muted hover:border-danger hover:text-danger"
            >
              <RotateCcw className="h-3.5 w-3.5" /> Factory Reset
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-3.5 w-3.5 text-danger" />
              <span className="text-[10px] text-danger">Delete ALL data?</span>
              <button
                onClick={handleReset}
                disabled={resetting}
                className="rounded-sm bg-danger px-3 py-1.5 text-[10px] font-semibold text-white hover:bg-danger/80"
              >
                {resetting ? "Resetting..." : "Yes, Reset"}
              </button>
              <button
                onClick={() => setResetConfirm(false)}
                className="rounded-sm border border-border px-3 py-1.5 text-[10px] text-muted hover:text-foreground"
              >
                Cancel
              </button>
            </div>
          )}
        </div>
        <p className="mt-2 text-[10px] text-muted">
          Re-run wizard: keeps data, lets you reconfigure storage/provider/model.
          Factory reset: deletes database, credentials, and auth — starts fresh.
        </p>
      </div>

      {/* Editable Setup Values */}
      <div className="mb-6 rounded-sm border border-border bg-card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Edit3 className="h-3.5 w-3.5 text-primary" />
          <p className="text-xs font-semibold text-muted">Setup Configuration</p>
        </div>
        <p className="text-[9px] text-muted mb-3">All values from setup wizard. Click a value to edit.</p>
        <div className="space-y-2">
          {Object.entries(setupValues).map(([label, value]) => {
            const editable = ["Interface Mode", "Storage Mode", "Default Model", "Max RPM", "Max Tool Iterations"].includes(label);
            const isEditing = editingField === label;
            return (
              <div key={label} className="flex items-center justify-between border-b border-border/30 pb-1.5 gap-2">
                <span className="text-[10px] text-muted flex-shrink-0">{label}</span>
                <div className="flex items-center gap-1.5 min-w-0">
                  {isEditing ? (
                    <>
                      <input
                        value={editValue}
                        onChange={e => setEditValue(e.target.value)}
                        onKeyDown={e => { if (e.key === "Enter") saveSetupField(label, editValue); if (e.key === "Escape") setEditingField(null); }}
                        autoFocus
                        className="w-36 rounded-sm border border-primary bg-background px-2 py-0.5 text-[10px] outline-none"
                      />
                      <button onClick={() => saveSetupField(label, editValue)} disabled={savingField === label}
                        className="text-primary hover:text-primary-hover">
                        <Save className="h-3 w-3" />
                      </button>
                    </>
                  ) : (
                    <>
                      <span className={`text-[10px] font-medium truncate max-w-[180px] ${saveSuccess === label ? "text-success" : "text-foreground"}`}>
                        {saveSuccess === label ? "Saved!" : value || "(empty)"}
                      </span>
                      {editable && (
                        <button onClick={() => { setEditingField(label); setEditValue(value === "(not set)" ? "" : value); }}
                          className="text-muted hover:text-primary flex-shrink-0">
                          <Edit3 className="h-2.5 w-2.5" />
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Config */}
      {cfg && (
        <div className="mb-6 rounded-sm border border-border bg-card p-4">
          <p className="mb-3 text-xs font-semibold text-muted">Configuration</p>
          <div className="space-y-2 text-xs">
            {Object.entries(cfg).map(([key, value]) => (
              <div key={key} className="flex justify-between border-b border-border/50 pb-1">
                <span className="text-muted">{key}</span>
                <span className="text-foreground font-medium">
                  {typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Logs */}
      <div className="rounded-sm border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-semibold text-muted">Recent Logs</p>
          <button onClick={() => system.logs(50).then(r => setLogs(r.logs || []))}
            className="text-[10px] text-primary hover:underline">Refresh</button>
        </div>
        <div className="max-h-80 overflow-y-auto rounded-sm bg-background p-3">
          {logs.length > 0 ? (
            <pre className="text-[10px] leading-relaxed text-muted whitespace-pre-wrap">
              {logs.join("\n")}
            </pre>
          ) : (
            <p className="text-xs text-muted text-center py-4">No logs available</p>
          )}
        </div>
      </div>
    </div>
  );
}
