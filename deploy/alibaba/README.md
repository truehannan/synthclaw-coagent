![AlibabaCloud](https://nexus.ingroupe.com/wp-content/uploads/2020/03/alibaba-cloud-logo.png)

# Alibaba Cloud Deployment

> Deployment documentation for SynthClaw CoAgent running on Alibaba Cloud Simple Application Server.

---

# Overview

SynthClaw CoAgent is deployed on **Alibaba Cloud Simple Application Server (SAS)** as a complete production-ready environment consisting of both the web frontend and the FastAPI backend.

This deployment demonstrates how SynthClaw can be self-hosted while providing collaborative AI agents through a modern web interface and CLI.

The current deployment uses **Qwen** as its active AI provider while remaining provider-agnostic through SynthClaw's unified provider runtime.

---

# Why Alibaba Cloud

Alibaba Cloud provides a lightweight and reliable environment for deploying SynthClaw's collaborative agent backend while maintaining low operational overhead.

The deployment currently hosts:

- React frontend
- FastAPI backend
- Agent Society runtime
- Qwen provider
- Health monitoring endpoints

This architecture can later be extended to support multiple instances, load balancing and distributed execution.

---

# Deployment Architecture

```text
                        User

              Web Interface / CLI

                    │
                    │
          HTTP / HTTPS Requests
                    │
                    ▼

──────────────────────────────────────────
Alibaba Cloud Simple Application Server
──────────────────────────────────────────

        ┌──────────────────────┐
        │        Nginx         │
        └──────────┬───────────┘
                   │
      ┌────────────┴────────────┐
      ▼                         ▼

 React Frontend          FastAPI Backend

                                 │

                         Agent Society Runtime

                                 │

          Planning • Memory • Delegation • Skills

                                 │

                   Multi Provider Runtime

                                 │

      Qwen (Active) • OpenAI • Anthropic • Gemini

 Workers AI • GitHub Models • NVIDIA • Hugging Face

        DigitalOcean • OpenRouter
```

---

# Components

## Frontend

Framework

- React

Served by

- Nginx

Responsibilities

- User interface
- Session management
- Agent visualization
- CLI-inspired dashboard
- Provider selection

---

## Backend

Framework

- FastAPI

Responsibilities

- Request processing
- Agent orchestration
- Task execution
- Memory
- Provider routing
- Skills
- MCP
- API endpoints

---

# AI Providers

SynthClaw supports multiple providers through a unified runtime.

Current providers include:

- Qwen ✅ (Active deployment provider)
- OpenAI
- Anthropic
- Gemini
- OpenRouter
- GitHub Models
- Workers AI
- NVIDIA
- Hugging Face
- DigitalOcean AI

Changing providers does not require backend architectural changes.

---

# Deployment Flow

```
Browser

↓

React

↓

FastAPI

↓

Agent Society

↓

Selected AI Provider

↓

Response

↓

Browser
```

The CLI follows the same execution path and communicates with the deployed backend.

---

# Deployment Process

The deployment is hosted on:

**Alibaba Cloud Simple Application Server**

Deployment workflow:

1. Source code is maintained on GitHub.
2. Updates are deployed through GitHub Actions.
3. FastAPI backend is started on the server.
4. Nginx serves the React frontend.
5. Health endpoints verify successful deployment.
6. Users can access the application through the public server IP.

---

# Health Verification

Health endpoint

```
GET /health
```

Example response

```json
{
    "service": "SynthClaw CoAgent",
    "status": "healthy",
    "provider": "Qwen",
    "deployment": "Alibaba Cloud Simple Application Server"
}
```

---

# Verification

The deployment can be verified through:

- Alibaba Cloud Simple Application Server console
- Running instance status
- Public frontend
- Health endpoint
- CLI communication with deployed backend
- GitHub deployment workflow

---

# Repository

Main project

```
https://github.com/truehannan/synthclaw-coagent
```

Deployment documentation

```
deploy/alibaba/
```

---

# Notes

SynthClaw CoAgent is designed with a deployment-agnostic architecture. While this deployment uses Alibaba Cloud Simple Application Server for the Qwen Cloud Hackathon, the collaborative Agent Society runtime remains portable across different cloud providers without requiring architectural changes.
