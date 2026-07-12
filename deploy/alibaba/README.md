# Alibaba Cloud ECS — Production Deployment

## Deployed Instance

| | |
|---|---|
| **Provider** | Alibaba Cloud (Elastic Compute Service) |
| **Region** | China East (Shanghai) / International |
| **Instance** | ecs.t6-c1m1.large (2 vCPU, 2GB RAM) |
| **OS** | Ubuntu 22.04 LTS |
| **Access** | `http://<ECS_IP>` (frontend + API) |

## Architecture on Alibaba Cloud

```
┌─────────────────────────────────────────────────────┐
│  Alibaba Cloud ECS Instance                         │
│                                                     │
│  ┌─────────────┐     ┌──────────────────────┐      │
│  │   nginx     │────▶│  Frontend (React)    │      │
│  │  :80        │     │  /opt/synthclaw/     │      │
│  │             │     │  frontend/dist/      │      │
│  │  /api/* ────│────▶│  API Server (FastAPI)│      │
│  │             │     │  :8000               │      │
│  └─────────────┘     └──────────────────────┘      │
│                              │                      │
│                       ┌──────▼──────┐               │
│                       │ Agent Core  │               │
│                       │ (Python)    │               │
│                       │ + Society   │               │
│                       └─────────────┘               │
│                                                     │
│  Security Group: 80 (HTTP), 443 (HTTPS)             │
└─────────────────────────────────────────────────────┘
```

## CI/CD Flow

```
GitHub Push (main) → GitHub Actions → SSH to ECS → Deploy + Restart
```

Pipeline steps:
1. Build & test locally (Python validation + frontend build)
2. SSH via `sshpass` using `secrets.HOST` + `secrets.PASSWORD`
3. Upload Python agent files + built frontend
4. Install deps in venv (`/opt/synthclaw/venv/`)
5. Configure nginx (reverse proxy :80 → frontend + API)
6. Create/restart systemd service
7. Health check with 5x retry

## Secrets Required

| Secret | Description |
|--------|-------------|
| `HOST` | ECS instance public IP |
| `PASSWORD` | SSH root password |

## Service Management

```bash
# Status
systemctl status synthclaw

# Logs  
journalctl -u synthclaw -f

# Restart
systemctl restart synthclaw

# Nginx
systemctl status nginx
```

## Proof of Deployment

After successful CI/CD run:
- **Frontend**: `http://<ECS_IP>/` — React web interface
- **API Health**: `http://<ECS_IP>/api/system/health` → `{"status": "ok"}`
- **Chat API**: `http://<ECS_IP>/api/chat/send` (POST with auth)
- **Agent Society**: `http://<ECS_IP>/api/society/status`
