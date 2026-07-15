import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { auth, setToken } from "@/lib/api";
import Mascot from "@/components/Mascot";

export default function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const res = await auth.login(password);
      if (res.token) {
        setToken(res.token);
        navigate("/");
      }
    } catch (err: any) {
      setError(err.message || "Invalid password");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <Mascot className="mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-primary">[+] Conclave</h1>
          <p className="mt-2 text-xs text-muted">Agent Society — Web Interface</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="mb-1 block text-xs text-muted">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password..."
              className="w-full rounded-sm border border-border bg-card px-3 py-2.5 text-sm text-foreground placeholder-muted/50 outline-none focus:border-primary"
              autoFocus
            />
          </div>

          {error && <p className="text-xs text-danger">{error}</p>}

          <button
            type="submit"
            disabled={loading || !password}
            className="w-full rounded-sm bg-primary px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-primary-hover disabled:opacity-50"
          >
            {loading ? "Authenticating..." : "Login"}
          </button>
        </form>

        <p className="text-center text-[10px] text-muted">
          Enter the password you created during signup.
        </p>
      </div>
    </div>
  );
}
