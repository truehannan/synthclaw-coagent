import { Outlet, NavLink, useLocation } from "react-router-dom";
import {
  MessageSquare, LayoutDashboard, Server, Brain, Key,
  BookOpen, Puzzle, Settings, LogOut,
} from "lucide-react";
import { clearToken } from "@/lib/api";
import SessionSidebar from "@/components/SessionSidebar";
import Mascot from "@/components/Mascot";

const NAV = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/chat", icon: MessageSquare, label: "Chat" },
  { to: "/providers", icon: Server, label: "Providers" },
  { to: "/society", icon: Brain, label: "Society" },
  { to: "/memory", icon: BookOpen, label: "Memory" },
  { to: "/credentials", icon: Key, label: "Credentials" },
  { to: "/skills", icon: Puzzle, label: "Skills" },
  { to: "/settings", icon: Settings, label: "Settings" },
];

export default function Layout() {
  const location = useLocation();

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="flex w-56 flex-col border-r border-border bg-card">
        {/* Brand */}
        <div className="flex items-center gap-2 border-b border-border px-4 py-3">
          <Mascot className="text-[5px] leading-[1]" />
          <span className="text-xs font-semibold">SynthClaw</span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-2 py-3">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-sm px-3 py-2 text-xs font-medium transition-colors ${
                  isActive
                    ? "bg-primary-dim text-primary"
                    : "text-muted hover:bg-card-hover hover:text-foreground"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <SessionSidebar />
        <div className="border-t border-border p-3">
          <button
            onClick={() => { clearToken(); window.location.href = "/login"; }}
            className="flex w-full items-center gap-2 rounded-sm px-3 py-2 text-xs text-muted hover:bg-card-hover hover:text-danger"
          >
            <LogOut className="h-3.5 w-3.5" />
            Logout
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
