#!/bin/bash
# SynthClaw-CoAgent — Server Setup Script
# Run this on a fresh Ubuntu/Debian VPS to set up the agent.
# Usage: bash setup_server.sh
set -e

INSTALL_DIR="${SYNTHCLAW_BASE_DIR:-/opt/agent}"

echo "╔═══════════════════════════════════════════════════╗"
echo "║     SynthClaw-CoAgent — Server Setup              ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

echo "=== [1/6] System update ==="
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git curl

echo "=== [2/6] Create ${INSTALL_DIR} structure ==="
mkdir -p "${INSTALL_DIR}/workspace"
cd "${INSTALL_DIR}"

echo "=== [3/6] Python virtual env + dependencies ==="
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade pip
"${INSTALL_DIR}/venv/bin/pip" install --quiet -r "${INSTALL_DIR}/requirements.txt"

echo "=== [4/6] Permissions ==="
chmod 700 "${INSTALL_DIR}"
chmod 755 "${INSTALL_DIR}/workspace"
if [ -f "${INSTALL_DIR}/.env" ]; then
    chmod 600 "${INSTALL_DIR}/.env"
fi

echo "=== [5/6] Systemd service ==="
cat > /etc/systemd/system/agent.service << SERVICE
[Unit]
Description=SynthClaw-CoAgent — Personal AI Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/venv/bin/python ${INSTALL_DIR}/main.py
Restart=always
RestartSec=10
StandardOutput=append:${INSTALL_DIR}/agent.log
StandardError=append:${INSTALL_DIR}/agent.log
EnvironmentFile=${INSTALL_DIR}/.env

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable agent.service

echo "=== [6/6] Done! ==="
echo ""
echo "  Next steps:"
echo "  1. Make sure .env is configured (run setup_cli.py or copy .env.example)"
echo "  2. Start:  systemctl start agent"
echo "  3. Logs:   journalctl -u agent -f"
echo "             tail -f ${INSTALL_DIR}/agent.log"
echo ""
echo "  For WhatsApp, ensure port ${WHATSAPP_PORT:-8443} is open and"
echo "  configure your Meta webhook URL to: https://your-domain/webhook"
