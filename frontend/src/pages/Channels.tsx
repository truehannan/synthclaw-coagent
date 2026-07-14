import { useEffect, useState } from "react";
import { system, memory, credentials } from "@/lib/api";
import { MessageCircle, Send as SendIcon, Check, Loader2, Save } from "lucide-react";

interface ChannelConfig {
  telegram_token: string;
  whatsapp_token: string;
  whatsapp_phone_number_id: string;
  whatsapp_verify_token: string;
  whatsapp_port: string;
  interface_mode: string;
}

export default function Channels() {
  const [config, setConfig] = useState<ChannelConfig>({
    telegram_token: "",
    whatsapp_token: "",
    whatsapp_phone_number_id: "",
    whatsapp_verify_token: "",
    whatsapp_port: "8443",
    interface_mode: "cli",
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState("");
  const [saved, setSaved] = useState("");

  useEffect(() => { loadConfig(); }, []);

  async function loadConfig() {
    try {
      const res = await system.config();
      setConfig(prev => ({
        ...prev,
        interface_mode: res.interface_mode || "cli",
      }));
      // Load channel tokens from memory/credentials
      const memRes = await memory.all();
      const facts = memRes.facts || {};
      setConfig(prev => ({
        ...prev,
        whatsapp_phone_number_id: facts.whatsapp_phone_number_id || "",
        whatsapp_verify_token: facts.whatsapp_verify_token || "",
        whatsapp_port: facts.whatsapp_port || "8443",
      }));
    } catch {}
    setLoading(false);
  }

  async function saveTelegram() {
    setSaving("telegram");
    try {
      await credentials.store("TELEGRAM_TOKEN", config.telegram_token, "Telegram Bot Token");
      await system.updateConfig("interface_mode", config.interface_mode.includes("telegram") ? config.interface_mode : "telegram");
      setSaved("telegram");
      setTimeout(() => setSaved(""), 2000);
    } catch {}
    setSaving("");
  }

  async function saveWhatsApp() {
    setSaving("whatsapp");
    try {
      await credentials.store("WHATSAPP_TOKEN", config.whatsapp_token, "WhatsApp Access Token");
      await memory.set("whatsapp_phone_number_id", config.whatsapp_phone_number_id);
      await memory.set("whatsapp_verify_token", config.whatsapp_verify_token);
      await memory.set("whatsapp_port", config.whatsapp_port);
      await system.updateConfig("interface_mode", config.interface_mode.includes("whatsapp") ? config.interface_mode : "whatsapp");
      setSaved("whatsapp");
      setTimeout(() => setSaved(""), 2000);
    } catch {}
    setSaving("");
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="p-6">
      <h1 className="text-lg font-bold">[+] Channels</h1>
      <p className="mb-6 text-xs text-muted">Connect messaging platforms to SynthClaw</p>

      {/* Current mode */}
      <div className="mb-6 rounded-sm border border-primary/20 bg-primary-dim px-4 py-2 text-xs">
        Current mode: <span className="font-semibold text-primary">{config.interface_mode}</span>
      </div>

      {/* Telegram */}
      <div className="mb-6 rounded-sm border border-border bg-card p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="h-8 w-8 rounded-full bg-blue-500/10 flex items-center justify-center">
            <SendIcon className="h-4 w-4 text-blue-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold">Telegram</h2>
            <p className="text-[10px] text-muted">Connect a Telegram bot (from @BotFather)</p>
          </div>
          {saved === "telegram" && <Check className="h-4 w-4 text-success ml-auto" />}
        </div>
        <div className="space-y-3">
          <div>
            <label className="text-[10px] text-muted block mb-1">Bot Token</label>
            <input type="password" value={config.telegram_token}
              onChange={e => setConfig(p => ({ ...p, telegram_token: e.target.value }))}
              placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
              className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
          </div>
          <button onClick={saveTelegram} disabled={!config.telegram_token || saving === "telegram"}
            className="flex items-center gap-2 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
            {saving === "telegram" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
            Save Telegram
          </button>
        </div>
      </div>

      {/* WhatsApp */}
      <div className="rounded-sm border border-border bg-card p-5">
        <div className="flex items-center gap-3 mb-4">
          <div className="h-8 w-8 rounded-full bg-green-500/10 flex items-center justify-center">
            <MessageCircle className="h-4 w-4 text-green-400" />
          </div>
          <div>
            <h2 className="text-sm font-bold">WhatsApp</h2>
            <p className="text-[10px] text-muted">Connect WhatsApp Business API</p>
          </div>
          {saved === "whatsapp" && <Check className="h-4 w-4 text-success ml-auto" />}
        </div>
        <div className="space-y-3">
          <div>
            <label className="text-[10px] text-muted block mb-1">Access Token</label>
            <input type="password" value={config.whatsapp_token}
              onChange={e => setConfig(p => ({ ...p, whatsapp_token: e.target.value }))}
              placeholder="WhatsApp Business API access token"
              className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
          </div>
          <div>
            <label className="text-[10px] text-muted block mb-1">Phone Number ID</label>
            <input value={config.whatsapp_phone_number_id}
              onChange={e => setConfig(p => ({ ...p, whatsapp_phone_number_id: e.target.value }))}
              placeholder="15551234567"
              className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-[10px] text-muted block mb-1">Verify Token</label>
              <input value={config.whatsapp_verify_token}
                onChange={e => setConfig(p => ({ ...p, whatsapp_verify_token: e.target.value }))}
                placeholder="Auto-generated"
                className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
            </div>
            <div>
              <label className="text-[10px] text-muted block mb-1">Port</label>
              <input value={config.whatsapp_port}
                onChange={e => setConfig(p => ({ ...p, whatsapp_port: e.target.value }))}
                placeholder="8443"
                className="w-full rounded-sm border border-border bg-background px-3 py-2 text-xs outline-none focus:border-primary" />
            </div>
          </div>
          <button onClick={saveWhatsApp} disabled={!config.whatsapp_token || saving === "whatsapp"}
            className="flex items-center gap-2 rounded-sm bg-primary px-4 py-2 text-xs font-semibold text-white hover:bg-primary-hover disabled:opacity-50">
            {saving === "whatsapp" ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
            Save WhatsApp
          </button>
        </div>
      </div>

      <p className="mt-4 text-[9px] text-muted">
        After saving, restart the service for channel changes to take effect.
        Use the Dashboard "Run Command" to run: <code className="text-primary">systemctl restart synthclaw</code>
      </p>
    </div>
  );
}
