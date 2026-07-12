import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { setToken } from "@/lib/api";

export default function Signup() {
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (password !== confirm) { setError("Passwords don't match"); return; }
    if (password.length < 4) { setError("Minimum 4 characters"); return; }
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const data = await res.json();
      if (data.token) { setToken(data.token); navigate("/"); }
      else { setError(data.detail || "Signup failed"); }
    } catch (err: any) { setError(err.message); }
    finally { setLoading(false); }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-primary">[+] SynthClaw</h1>
          <p className="mt-2 text-xs text-muted">First time setup — create your password</p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="mb-1 block text-xs text-muted">Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
              placeholder="Create a password..." className="w-full rounded-sm border border-border bg-card px-3 py-2.5 text-sm text-foreground placeholder-muted/50 outline-none focus:border-primary" autoFocus />
          </div>
          <div>
            <label className="mb-1 block text-xs text-muted">Confirm Password</label>
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)}
              placeholder="Confirm..." className="w-full rounded-sm border border-border bg-card px-3 py-2.5 text-sm text-foreground placeholder-muted/50 outline-none focus:border-primary" />
          </div>
          {error && <p className="text-xs text-danger">{error}</p>}
          <button type="submit" disabled={loading || !password}
            className="w-full rounded-sm bg-primary px-4 py-2.5 text-sm font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
            {loading ? "Creating..." : "Create Account"}
          </button>
        </form>
        <p className="text-center text-[10px] text-muted">This password is used to access the web interface.</p>
      </div>
    </div>
  );
}
