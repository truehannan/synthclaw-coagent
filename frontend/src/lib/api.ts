const BASE = "/api";

let _token = localStorage.getItem("synthclaw_token") || "";

export function setToken(token: string) {
  _token = token;
  localStorage.setItem("synthclaw_token", token);
}

export function getToken() { return _token; }
export function clearToken() { _token = ""; localStorage.removeItem("synthclaw_token"); }

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
    clearToken();
    window.location.href = "/login";
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
  login: (password: string) => request("/auth/login", { method: "POST", body: JSON.stringify({ password }) }),
};

// ── Chat ─────────────────────────────────────────────────────────────────────
export const chat = {
  history: (limit = 50) => request(`/chat/history?limit=${limit}`),
  clear: () => request("/chat/clear", { method: "POST" }),
  stop: () => request("/chat/stop", { method: "POST" }),
  approve: () => request("/chat/approve", { method: "POST" }),
  deny: () => request("/chat/deny", { method: "POST" }),
  taskStatus: () => request("/chat/task-status"),
  /** Send message with SSE streaming */
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
};
