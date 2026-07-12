import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { getToken } from "@/lib/api";
import Layout from "@/components/Layout";
import Login from "@/pages/Login";
import Signup from "@/pages/Signup";
import Chat from "@/pages/Chat";
import Dashboard from "@/pages/Dashboard";
import Providers from "@/pages/Providers";
import Memory from "@/pages/Memory";
import Credentials from "@/pages/Credentials";
import Skills from "@/pages/Skills";
import Society from "@/pages/Society";
import Settings from "@/pages/Settings";

function AuthGate({ children }: { children: React.ReactNode }) {
  const token = getToken();
  const [checking, setChecking] = useState(true);
  const [needsSignup, setNeedsSignup] = useState(false);

  useEffect(() => {
    // If already has token, skip check
    if (token) { setChecking(false); return; }
    // Check if user exists
    fetch("/api/auth/exists").then(r => r.json()).then(data => {
      if (!data.exists) setNeedsSignup(true);
      setChecking(false);
    }).catch(() => setChecking(false));
  }, []);

  if (checking) return <div className="flex min-h-screen items-center justify-center bg-background"><p className="text-muted text-xs">Loading...</p></div>;
  if (!token && needsSignup) return <Navigate to="/signup" replace />;
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/" element={<AuthGate><Layout /></AuthGate>}>
        <Route index element={<Dashboard />} />
        <Route path="chat" element={<Chat />} />
        <Route path="providers" element={<Providers />} />
        <Route path="memory" element={<Memory />} />
        <Route path="credentials" element={<Credentials />} />
        <Route path="skills" element={<Skills />} />
        <Route path="society" element={<Society />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
