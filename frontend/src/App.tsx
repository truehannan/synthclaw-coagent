import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { getToken } from "@/lib/api";
import Layout from "@/components/Layout";
import Login from "@/pages/Login";
import Signup from "@/pages/Signup";
import SetupWizard from "@/pages/SetupWizard";
import Chat from "@/pages/Chat";
import Dashboard from "@/pages/Dashboard";
import Providers from "@/pages/Providers";
import Memory from "@/pages/Memory";
import Credentials from "@/pages/Credentials";
import Skills from "@/pages/Skills";
import Society from "@/pages/Society";
import Settings from "@/pages/Settings";
import Integrations from "@/pages/Integrations";

function AuthGate({ children }: { children: React.ReactNode }) {
  const token = getToken();
  const [checking, setChecking] = useState(true);
  const [needsSignup, setNeedsSignup] = useState(false);
  const [needsSetup, setNeedsSetup] = useState(false);

  useEffect(() => {
    // If already has token, check if provider is configured
    if (token) {
      fetch("/api/setup/status", {
        headers: { "X-API-Token": token },
      })
        .then((r) => r.json())
        .then((data) => {
          if (!data.configured) setNeedsSetup(true);
          setChecking(false);
        })
        .catch(() => setChecking(false));
      return;
    }
    // No token — check if user exists
    fetch("/api/auth/exists")
      .then((r) => r.json())
      .then((data) => {
        if (!data.exists) setNeedsSignup(true);
        setChecking(false);
      })
      .catch(() => setChecking(false));
  }, []);

  if (checking)
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <p className="text-muted text-xs">Loading...</p>
      </div>
    );
  if (!token && needsSignup) return <Navigate to="/signup" replace />;
  if (!token) return <Navigate to="/login" replace />;
  if (needsSetup) return <Navigate to="/setup" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/setup" element={<SetupWizard />} />
      <Route path="/" element={<AuthGate><Layout /></AuthGate>}>
        <Route index element={<Dashboard />} />
        <Route path="chat" element={<Chat />} />
        <Route path="chat/:id" element={<Chat />} />
        <Route path="providers" element={<Providers />} />
        <Route path="memory" element={<Memory />} />
        <Route path="credentials" element={<Credentials />} />
        <Route path="skills" element={<Skills />} />
        <Route path="integrations" element={<Integrations />} />
        <Route path="society" element={<Society />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
