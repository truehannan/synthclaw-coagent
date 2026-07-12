#!/bin/bash
# SynthClaw Alibaba Cloud ECS Deployment Script
# Usage: ./deploy.sh <HOST_IP> <SSH_PASSWORD> [SSH_USER]
#
# Prerequisites on the ECS instance:
# - Ubuntu 22.04+ (or Debian-based)
# - Port 3000 and 8000 open in security group

set -e

HOST="${1:?Usage: ./deploy.sh <HOST_IP> <SSH_PASSWORD> [SSH_USER]}"
PASSWORD="${2:?Password required}"
USER="${3:-root}"
REMOTE_DIR="/opt/synthclaw"

echo "╭─── SynthClaw Alibaba Cloud Deploy ───╮"
echo "│ Host: $HOST                           │"
echo "│ User: $USER                           │"
echo "│ Dir:  $REMOTE_DIR                     │"
echo "╰───────────────────────────────────────╯"
echo ""

# Check sshpass
if ! command -v sshpass &>/dev/null; then
    echo "Installing sshpass..."
    apt-get install -y sshpass 2>/dev/null || brew install sshpass 2>/dev/null || {
        echo "ERROR: sshpass not found. Install it first."
        exit 1
    }
fi

SSH="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no $USER@$HOST"
SCP="sshpass -p '$PASSWORD' scp -o StrictHostKeyChecking=no"

echo "[1/6] Creating remote directory..."
eval $SSH "mkdir -p $REMOTE_DIR"

echo "[2/6] Uploading files..."
eval $SCP -r \
    main.py agent.py agents.py tools.py memory.py config.py \
    model_fetcher.py d1_storage.py api_server.py requirements.txt \
    "$USER@$HOST:$REMOTE_DIR/"

echo "[3/6] Installing Python dependencies..."
eval $SSH "cd $REMOTE_DIR && pip3 install -r requirements.txt -q"

echo "[4/6] Writing .env..."
eval $SSH "cat > $REMOTE_DIR/.env << 'ENVEOF'
SYNTHCLAW_BASE_DIR=$REMOTE_DIR
SYNTHCLAW_API_PORT=8000
SYNTHCLAW_API_HOST=0.0.0.0
INTERFACE_MODE=cli
ENVEOF
chmod 600 $REMOTE_DIR/.env"

echo "[5/6] Creating systemd service..."
eval $SSH "cat > /etc/systemd/system/synthclaw.service << 'SVCEOF'
[Unit]
Description=SynthClaw Agent Society
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REMOTE_DIR
ExecStart=/usr/bin/python3 $REMOTE_DIR/main.py
Restart=always
RestartSec=5
Environment=SYNTHCLAW_BASE_DIR=$REMOTE_DIR

[Install]
WantedBy=multi-user.target
SVCEOF
systemctl daemon-reload && systemctl enable synthclaw"

echo "[6/6] Starting service..."
eval $SSH "systemctl restart synthclaw"
sleep 3

# Health check
echo ""
if eval $SSH "curl -sf http://localhost:8000/api/system/health" >/dev/null 2>&1; then
    echo "✓ SynthClaw is running on $HOST:8000"
    echo "  API:    http://$HOST:8000"
    echo "  Health: http://$HOST:8000/api/system/health"
else
    echo "⚠ Service started but health check failed"
    echo "  Check: ssh $USER@$HOST 'journalctl -u synthclaw -n 20'"
fi
echo ""
