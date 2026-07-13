import { useEffect, useState } from "react";
import { system, auth } from "@/lib/api";
import { KeyRound, Check } from "lucide-react";

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

  useEffect(() => {
    system.config().then(setCfg).catch(() => {});
    system.logs(30).then((r) => setLogs(r.logs || [])).catch(() => {});
  }, []);

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
