<p align="center"><img src=/public/icon.png alt=synthclaw style="length: 100px; width: 100px;"/></p>

<h1 align="center">SynthClaw CoAgent</h1>

**Your personal AI agent with full server control, 50 tools, 8 LLM providers, 1000+ app integrations, and an interactive CLI dashboard.**
      
![Size](https://img.shields.io/github/repo-size/truehannan/synthclaw-coagent?style=for-the-badge&color=8b5cf6&labelColor=1B1B1B)![Language](https://img.shields.io/github/languages/top/truehannan/synthclaw-coagent?style=for-the-badge&color=00bf63&labelColor=1B1B1B)![Stars](https://img.shields.io/github/stars/truehannan/synthclaw-coagent?style=for-the-badge&color=00bf63&labelColor=1B1B1B)![Contributors](https://img.shields.io/github/contributors/truehannan/synthclaw-coagent?style=for-the-badge&color=00bf63&labelColor=1B1B1B) <a href="https://twitter.com/intent/follow?screen_name=truehannan"><img src="https://img.shields.io/twitter/follow/truehannan.svg?label=Follow%20@truehannan" alt="Follow @truehannan" /></a> <a href="https://github.com/truehannan/synthclaw-coagent/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Medusa is released under the MIT license." /></a>

<p align="center"><a href="https://www.producthunt.com/products/synthclaw-coagent?embed=true&amp;utm_source=badge-featured&amp;utm_medium=badge&amp;utm_campaign=badge-synthclaw-coagent" target="_blank" rel="noopener noreferrer"><img alt="SynthClaw CoAgent - Synthclaw is alternative to OpenClaw with no missed feature | Product Hunt" width="250" height="54" src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1186019&amp;theme=light&amp;t=1782968063484"></a></p>

---

## Quick Start

```bash
git clone https://github.com/truehannan/synthclaw-coagent.git
cd synthclaw-coagent
npm install && npm run link
synthclaw setup       # interactive wizard
synthclaw start       # starts the agent
synthclaw agent       # interactive dashboard
```

---

## What It Does

- Runs on a $6/mo VPS, talks to you via **Telegram**, **WhatsApp**, or **CLI**
- Executes shell commands, manages files, deploys services, calls APIs
- **50 built-in tools** — shell, file I/O, HTTP, systemd, search, code execution, reminders
- **8 LLM providers** — DigitalOcean, OpenAI, Anthropic, Google, NVIDIA, HuggingFace, OpenRouter, GitHub
- **1000+ app integrations** via Composio (Gmail, Slack, GitHub, Notion, etc.)
- **Dynamic API registration** — give it a credential, it auto-detects the service and starts calling it
- **MCP support** — paste MCP JSON configs and the agent can use those tool servers
- **Skills system** — upload Claude-format skills (.zip) or install by name
- **In-memory code execution** — runs Node.js/Python/Bash without writing files
- **Auto-resumes on failure** — task state saved to SQLite, picks up where it left off

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `synthclaw setup` | Interactive wizard (edit mode if already configured) |
| `synthclaw start` | Start the agent (auto-creates venv, installs deps) |
| `synthclaw stop` | Stop the agent |
| `synthclaw agent` | **Interactive dashboard** — chat REPL with system meters |
| `synthclaw agent "task"` | Autonomous execution mode |
| `synthclaw status` | Show agent status + health |
| `synthclaw logs -f` | Tail logs |
| `synthclaw model <name>` | Switch LLM model |
| `synthclaw models` | List all available models |
| `synthclaw deploy` | Deploy to remote VPS |

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Register as owner |
| `/model`, `/models` | Switch/list models |
| `/run <cmd>` | Execute shell command |
| `/plan <task>` | Plan without executing |
| `/agent <task>` | Autonomous execution |
| `/skills` | Manage skills |
| `/apis` | Manage registered APIs |
| `/connectors` | Composio app connections (1000+ apps) |
| `/mcp` | Manage MCP servers |
| `/memory`, `/creds` | Facts and credentials |
| `/usage` | Token usage & costs |

---

## Providers (8)

| Provider | Prefix | Models |
|----------|--------|--------|
| DigitalOcean | _(default)_ | Llama 3.3, Mistral, DeepSeek, Qwen, Claude, GPT-4o |
| OpenAI | `openai-` | GPT-4o, GPT-4.1, o3 |
| Anthropic | `anthropic-` | Claude Sonnet 4, Opus 4 |
| Google | `google:` | Gemini 2.5 Flash/Pro |
| NVIDIA | `nvidia:` | Llama 405B, Nemotron, DeepSeek R1 |
| HuggingFace | `hf:` | Llama, Qwen, Mistral, Phi-4 |
| OpenRouter | `openrouter:` | Any model via proxy |
| GitHub | `github:` | GPT-4o, Llama, Mistral |

---

## Key Features

### Dynamic API Registration
Give the agent a credential — it auto-detects the service and registers it:
```
You: "Store my stripe key: sk_live_xxx"
Agent: Stored + auto-registered Stripe API (9 endpoints)
Later: "Show my charges" → agent calls Stripe API directly
```
18 services auto-detected: Stripe, Vercel, Cloudflare, GitHub, Notion, Airtable, etc.

### Composio (1000+ Apps)
```
/connectors connect github   → OAuth link
/connectors connect gmail    → OAuth link
```
Then the agent can send emails, create issues, post to Slack — all via natural language.

### MCP Support
```
/mcp add {"mcpServers":{"filesystem":{"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","."]}}}
```

### In-Memory Code Execution
The agent runs code directly without writing files:
```
exec_code(code="console.log(Math.PI * 5**2)", lang="node")
exec_code(code="import json; print(json.dumps({'x': 42}))", lang="python")
```

### Skills
```
/skills install coding-agent          # search + install from GitHub
/skills install https://url/skill.zip # direct URL
```
Or send a .zip file to the bot — auto-installs if it contains SKILL.md.

### Interactive CLI Dashboard
```bash
synthclaw agent
```
Launches a terminal dashboard with:
- System gauges (CPU/MEM/DISK)
- Server info (IP, uptime, model)
- Chat REPL with `/` command autocomplete
- Red theme throughout

---

## Setup Wizard

The wizard detects if you've already configured:
- **First run**: full setup (storage, interface, credentials, model, limits)
- **Re-run**: edit mode (Enter to keep current values)

Options include:
- **Storage**: Local SQLite or Cloudflare D1 + R2
- **RPM limiting**: cap requests per minute to control costs
- **Composio**: optional 1000+ app integration key

---

## Architecture

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Telegram   │  │   WhatsApp   │  │ CLI Dashboard│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       └────────────┬────┘───────────────────┘
                    │
             ┌──────▼──────┐
             │  Agent Core │ ← 8 LLM providers + tool loop
             └──────┬──────┘
                    │
     ┌──────────────┼──────────────┐
     │              │              │
┌────▼────┐  ┌─────▼─────┐  ┌────▼────┐
│ 50 Tools│  │  Memory   │  │Composio │
│exec_code│  │  SQLite   │  │ MCP     │
│api_call │  │  +Fernet  │  │ Skills  │
└─────────┘  └───────────┘  └─────────┘
```

[![Star History](https://api.star-history.com/svg?repos=truehannan/synthclaw-coagent&type=Date&theme=dark)](https://star-history.com/#truehannan/synthclaw-coagent&Date)

---

## License

Source Available — Non-Commercial. Free for personal use.

---

**Made by [@truehannan](https://github.com/truehannan)**
