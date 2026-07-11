import { Routes, Route, Navigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { getToken } from "@/lib/api";
import Layout from "@/components/Layout";
import Login from "@/pages/Login";
import Chat from "@/pages/Chat";
import Dashboard from "@/pages/Dashboard";
import Providers from "@/pages/Providers";
import Memory from "@/pages/Memory";
import Credentials from "@/pages/Credentials";
import Skills from "@/pages/Skills";
import Society from "@/pages/Society";
import Settings from "@/pages/Settings";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = getToken();
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
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
