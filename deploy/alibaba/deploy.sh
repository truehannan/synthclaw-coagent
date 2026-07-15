#!/bin/bash
# Conclave Alibaba Cloud ECS Deployment Script
# Usage: ./deploy.sh <HOST_IP> <SSH_PASSWORD> [SSH_USER]
#
# Prerequisites on the ECS instance:
# - Ubuntu 22.04+ (or Debian-based)
# - Port 3000 and 8000 open in security group

set -e

HOST="${1:?Usage: ./deploy.sh <HOST_IP> <SSH_PASSWORD> [SSH_USER]}"
PASSWORD="${2:?Password required}"
USER="${3:-root}"
REMOTE_DIR="/opt/conclave"

echo "╭─── Conclave Alibaba Cloud Deploy ───╮"
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
eval $SSH "apt-get update -qq && apt-get install -y -qq python3-venv python3-pip nginx curl > /dev/null 2>&1"
eval $SSH "cd $REMOTE_DIR && python3 -m venv venv && ./venv/bin/pip install --upgrade pip -q && ./venv/bin/pip install -r requirements.txt -q"

echo "[4/6] Writing .env..."
eval $SSH "cat > $REMOTE_DIR/.env << 'ENVEOF'
CONCLAVE_BASE_DIR=$REMOTE_DIR
CONCLAVE_API_PORT=8000
CONCLAVE_API_HOST=0.0.0.0
INTERFACE_MODE=cli
ENVEOF
chmod 600 $REMOTE_DIR/.env"

echo "[5/6] Creating systemd service..."
eval $SSH "cat > /etc/systemd/system/conclave.service << 'SVCEOF'
[Unit]
Description=Conclave Agent Society
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/venv/bin/python3 $REMOTE_DIR/main.py
Restart=always
RestartSec=5
Environment=CONCLAVE_BASE_DIR=$REMOTE_DIR

[Install]
WantedBy=multi-user.target
SVCEOF
systemctl daemon-reload && systemctl enable conclave"

echo "[6/6] Starting service..."
eval $SSH "systemctl restart conclave"
sleep 3

# Health check
echo ""
if eval $SSH "curl -sf http://localhost:8000/api/system/health" >/dev/null 2>&1; then
    echo "✓ Conclave is running on $HOST:8000"
    echo "  API:    http://$HOST:8000"
    echo "  Health: http://$HOST:8000/api/system/health"
else
    echo "⚠ Service started but health check failed"
    echo "  Check: ssh $USER@$HOST 'journalctl -u conclave -n 20'"
fi
echo ""
