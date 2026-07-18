const BASE = "/api";

let _token = localStorage.getItem("conclave_token") || "";

export function setToken(token: string) {
  _token = token;
  localStorage.setItem("conclave_token", token);
}

export function getToken() { return _token; }
export function clearToken() { _token = ""; localStorage.removeItem("conclave_token"); }

async function request<T = any>(path: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      "X-API-Token": _token,
      ...opts.headers,
    },
  });
  if (res.status === 401) {
    // Don't redirect if already on auth/setup pages (prevents loops)
    const loc = window.location.pathname;
    if (loc !== "/login" && loc !== "/signup" && loc !== "/setup") {
      clearToken();
      window.location.href = "/login";
    }
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Auth ─────────────────────────────────────────────────────────────────────
export const auth = {
  status: () => request("/auth/status"),
  exists: () => fetch(`${BASE}/auth/exists`).then(r => r.json()),
  signup: (password: string) => fetch(`${BASE}/auth/signup`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password }) }).then(r => r.json()),
  login: (password: string) => request("/auth/login", { method: "POST", body: JSON.stringify({ password }) }),
  changePassword: (current_password: string, new_password: string) => request("/auth/change-password", { method: "POST", body: JSON.stringify({ current_password, new_password }) }),
};

// ── Chat ─────────────────────────────────────────────────────────────────────
export const chat = {
  history: (limit = 50) => request(`/chat/history?limit=${limit}`),
  clear: () => request("/chat/clear", { method: "POST" }),
  stop: () => request("/chat/stop", { method: "POST" }),
  approve: () => request("/chat/approve", { method: "POST" }),
  deny: () => request("/chat/deny", { method: "POST" }),
  taskStatus: () => request("/chat/task-status"),
  /** Full agentic run with structured SSE events */
  run: (message: string, model?: string) => {
    return fetch(`${BASE}/chat/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Token": _token },
      body: JSON.stringify({ message, model }),
    });
  },
  /** Legacy: simple streaming (kept for fallback) */
  sendStream: (message: string, model?: string) => {
    return fetch(`${BASE}/chat/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Token": _token },
      body: JSON.stringify({ message, model }),
    });
  },
};

// ── Providers & Models ───────────────────────────────────────────────────────
export const providers = {
  list: () => request("/providers"),
  models: (name: string) => request(`/providers/${name}/models`),
  storeKey: (name: string, key: string) => request(`/providers/${name}/key`, { method: "POST", body: JSON.stringify({ key }) }),
  deleteKey: (name: string) => request(`/providers/${name}/key`, { method: "DELETE" }),
};

export const models = {
  all: () => request("/models"),
  current: () => request("/models/current"),
  switch: (model: string) => request("/models/switch", { method: "POST", body: JSON.stringify({ model }) }),
  usage: () => request("/models/usage"),
};

// ── Memory & Credentials ─────────────────────────────────────────────────────
export const memory = {
  all: () => request("/memory"),
  set: (key: string, value: string) => request("/memory", { method: "POST", body: JSON.stringify({ key, value }) }),
  del: (key: string) => request(`/memory/${key}`, { method: "DELETE" }),
};

export const credentials = {
  list: () => request("/credentials"),
  store: (name: string, value: string, description = "") => request("/credentials", { method: "POST", body: JSON.stringify({ name, value, description }) }),
  del: (name: string) => request(`/credentials/${name}`, { method: "DELETE" }),
};

// ── Skills ───────────────────────────────────────────────────────────────────
export const skills = {
  list: () => request("/skills"),
  install: (source: string) => request("/skills/install", { method: "POST", body: JSON.stringify({ source }) }),
  uninstall: (name: string) => request(`/skills/${name}`, { method: "DELETE" }),
  reinstall: () => request("/skills/reinstall", { method: "POST" }),
};

// ── System ───────────────────────────────────────────────────────────────────
export const system = {
  status: () => request("/system/status"),
  config: () => request("/system/config"),
  updateConfig: (key: string, value: string) => request("/system/config", { method: "POST", body: JSON.stringify({ key, value }) }),
  health: () => fetch(`${BASE}/system/health`).then(r => r.json()),
  logs: (lines = 50) => request(`/system/logs?lines=${lines}`),
  run: (command: string) => request("/system/run", { method: "POST", body: JSON.stringify({ command }) }),
};

// ── Society ──────────────────────────────────────────────────────────────────
export const society = {
  status: () => request("/society/status"),
  reset: () => request("/society/reset", { method: "POST" }),
};

// ── Sessions ─────────────────────────────────────────────────────────────────
export const sessions = {
  list: () => request("/sessions"),
  create: (name: string) => request("/sessions", { method: "POST", body: JSON.stringify({ name }) }),
  del: (id: string) => request(`/sessions/${id}`, { method: "DELETE" }),
  switch: (id: string) => request(`/sessions/${id}/switch`, { method: "POST" }),
};

// ── Integrations ─────────────────────────────────────────────────────────────
export const apis = {
  list: () => request("/apis"),
};

export const composio = {
  connections: () => request("/composio/connections"),
  tools: (page = 1, search = "", toolkit = "") => request(`/composio/tools?page=${page}&search=${encodeURIComponent(search)}&toolkit=${encodeURIComponent(toolkit)}`),
  connect: (toolkit: string, apiKey?: string) => request(`/composio/connect/${toolkit}`, { method: "POST", body: JSON.stringify(apiKey ? { api_key: apiKey } : {}) }),
  disconnect: (connectionId: string) => request(`/composio/disconnect/${connectionId}`, { method: "DELETE" }),
  triggers: {
    list: () => request("/composio/triggers"),
    create: (slug: string, config?: any, connectedAccountId?: string) => request("/composio/triggers", { method: "POST", body: JSON.stringify({ slug, config, connected_account_id: connectedAccountId }) }),
    del: (triggerId: string) => request(`/composio/triggers/${triggerId}`, { method: "DELETE" }),
  },
};

// ── MCP Servers ──────────────────────────────────────────────────────────────
export const mcp = {
  list: () => request("/mcp/servers"),
  add: (config: any) => request("/mcp/servers", { method: "POST", body: JSON.stringify({ config }) }),
  remove: (name: string) => request(`/mcp/servers/${name}`, { method: "DELETE" }),
};

// ── Setup ────────────────────────────────────────────────────────────────────
export const setup = {
  status: () => request("/setup/status"),
};
