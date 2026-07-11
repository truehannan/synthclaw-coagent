export interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  ts?: string;
}

export interface Provider {
  name: string;
  slug: string;
  emoji: string;
  configured: boolean;
  key_name: string;
}

export interface Session {
  id: string;
  name: string;
  created_at: number;
  last_active: number;
  chat_id: number;
}

export interface AgentNode {
  id: string;
  role: string;
  name: string;
  status: string;
  task: string;
  elapsed: number;
  children: AgentNode[];
}

export interface SystemStatus {
  hostname: string;
  platform: string;
  python: string;
  uptime: number;
  cpu_percent: number;
  memory: { total: number; used: number; percent: number };
  disk: { total: number; used: number; percent: number };
}
