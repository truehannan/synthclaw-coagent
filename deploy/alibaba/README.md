# Alibaba Cloud Deployment

Deploy SynthClaw to an Alibaba Cloud ECS instance.

## Prerequisites

1. **ECS Instance** — Ubuntu 22.04, 2GB+ RAM, Python 3.9+
2. **Security Group** — Open ports: 8000 (API), 3000 (Frontend, optional)
3. **SSH Access** — Root password or key-based auth

## Quick Deploy (Script)

```bash
cd deploy/alibaba
chmod +x deploy.sh
./deploy.sh <ECS_IP> <ROOT_PASSWORD>
```

This will:
- Upload all agent files to `/opt/synthclaw`
- Install Python dependencies
- Create systemd service
- Start the agent

## Docker Deploy

```bash
# Build image
docker build -t synthclaw -f deploy/alibaba/Dockerfile .

# Run container
docker run -d \
  --name synthclaw \
  -p 3000:3000 \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your-key-here \
  -e OPENAI_API_BASE=https://inference.do-ai.run/v1 \
  -e DEFAULT_MODEL=llama3.3-70b-instruct \
  synthclaw
```

## CI/CD (GitHub Actions)

The repository includes `.github/workflows/deploy.yml` which automatically deploys on push to main.

**Required Secrets:**
- `HOST` — ECS instance IP address
- `PASSWORD` — SSH root password
- `SSH_USER` — (optional, defaults to `root`)

## Post-Deployment

1. Set up your API key:
```bash
ssh root@<ECS_IP> "echo 'OPENAI_API_KEY=your-key' >> /opt/synthclaw/.env && systemctl restart synthclaw"
```

2. Access the API:
```
http://<ECS_IP>:8000/api/system/health
```

3. Connect Telegram:
```bash
ssh root@<ECS_IP> "echo 'TELEGRAM_TOKEN=your-token
INTERFACE_MODE=telegram' >> /opt/synthclaw/.env && systemctl restart synthclaw"
```

## Troubleshooting

```bash
# Check service status
ssh root@<IP> "systemctl status synthclaw"

# View logs
ssh root@<IP> "journalctl -u synthclaw -f"

# Restart
ssh root@<IP> "systemctl restart synthclaw"
```
