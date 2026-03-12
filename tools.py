"""
SynthClaw-CoAgent — Tool Implementations
12 tools the agent can invoke: shell, files, HTTP, services, credentials, memory.
"""
import json
import subprocess
import requests
from pathlib import Path
from memory import (
    store_credential, get_credential, list_credentials,
    set_memory, get_memory, get_all_memory,
)
from config import WORKSPACE_DIR


# ── Tool implementations ─────────────────────────────────────────────────────

def run_command(command: str, timeout: int = 30) -> dict:
    """Execute a shell command on the server."""
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(WORKSPACE_DIR),
        )
        return {
            "stdout": r.stdout[-3000:],
            "stderr": r.stderr[-1000:],
            "returncode": r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Timed out after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"error": str(e), "returncode": -1}


def write_file(path: str, content: str) -> dict:
    """Write content to a file (relative to workspace unless absolute)."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return {"success": True, "path": str(p)}
    except Exception as e:
        return {"error": str(e)}


def read_file(path: str) -> dict:
    """Read a file's contents."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"Not found: {path}"}
        text = p.read_text()
        return {"content": text[:5000], "truncated": len(text) > 5000, "size": len(text)}
    except Exception as e:
        return {"error": str(e)}


def list_files(path: str = ".") -> dict:
    """List files/dirs in a path."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"Not found: {path}"}
        items = [
            {"name": i.name, "type": "dir" if i.is_dir() else "file",
             "size": i.stat().st_size if i.is_file() else None}
            for i in sorted(p.iterdir())
        ]
        return {"items": items, "path": str(p)}
    except Exception as e:
        return {"error": str(e)}


def http_request(url: str, method: str = "GET",
                  headers: dict = None, body: str = None,
                  timeout: int = 30) -> dict:
    """Make an HTTP request."""
    try:
        resp = requests.request(
            method.upper(), url,
            headers=headers or {}, data=body, timeout=timeout,
        )
        return {
            "status_code": resp.status_code,
            "body": resp.text[:3000],
            "headers": dict(resp.headers),
        }
    except Exception as e:
        return {"error": str(e)}


def spawn_service(name: str, command: str, description: str = "") -> dict:
    """Create and start a persistent systemd service."""
    try:
        unit = f"""[Unit]
Description={description or name}
After=network.target

[Service]
Type=simple
WorkingDirectory={WORKSPACE_DIR}
ExecStart={command}
Restart=always
RestartSec=5
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
"""
        Path(f"/etc/systemd/system/{name}.service").write_text(unit)
        r = subprocess.run(
            f"systemctl daemon-reload && systemctl enable {name} && systemctl start {name}",
            shell=True, capture_output=True, text=True,
        )
        return {
            "success": r.returncode == 0,
            "service": name,
            "output": (r.stdout + r.stderr)[-1000:],
        }
    except Exception as e:
        return {"error": str(e)}


def stop_service(name: str) -> dict:
    """Stop a systemd service."""
    try:
        r = subprocess.run(
            f"systemctl stop {name}", shell=True, capture_output=True, text=True
        )
        return {"success": r.returncode == 0, "output": r.stdout + r.stderr}
    except Exception as e:
        return {"error": str(e)}


def service_status(name: str = None) -> dict:
    """Get service status."""
    try:
        cmd = (
            f"systemctl status {name} --no-pager"
            if name
            else "systemctl list-units --type=service --state=running --no-pager"
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {"output": (r.stdout + r.stderr)[:3000]}
    except Exception as e:
        return {"error": str(e)}


def store_cred(name: str, value: str, description: str = "") -> dict:
    """Store an encrypted credential."""
    try:
        store_credential(name, value, description)
        return {"success": True, "stored": name}
    except Exception as e:
        return {"error": str(e)}


def get_cred(name: str) -> dict:
    """Retrieve a stored credential."""
    try:
        v = get_credential(name)
        if v is None:
            return {"error": f"'{name}' not found"}
        return {"name": name, "value": v}
    except Exception as e:
        return {"error": str(e)}


def remember(key: str, value: str) -> dict:
    """Store a persistent memory fact."""
    try:
        set_memory(key, value)
        return {"success": True, "key": key, "value": value}
    except Exception as e:
        return {"error": str(e)}


def recall(key: str = None) -> dict:
    """Recall a memory or all memories."""
    try:
        if key:
            return {"key": key, "value": get_memory(key)}
        return {"memory": get_all_memory()}
    except Exception as e:
        return {"error": str(e)}


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "run_command": {
        "fn": run_command,
        "description": "Execute a shell command on the server",
        "params": {"command": "str (required)", "timeout": "int (optional, default 30)"},
    },
    "write_file": {
        "fn": write_file,
        "description": "Write content to a file. Path is relative to workspace unless absolute.",
        "params": {"path": "str", "content": "str"},
    },
    "read_file": {
        "fn": read_file,
        "description": "Read a file's contents",
        "params": {"path": "str"},
    },
    "list_files": {
        "fn": list_files,
        "description": "List files in a directory",
        "params": {"path": "str (optional, default '.')"},
    },
    "http_request": {
        "fn": http_request,
        "description": "Make an HTTP request to any URL",
        "params": {
            "url": "str", "method": "GET/POST/PUT/DELETE (default GET)",
            "headers": "dict (optional)", "body": "str (optional)", "timeout": "int (optional)",
        },
    },
    "spawn_service": {
        "fn": spawn_service,
        "description": "Create & start a persistent background systemd service",
        "params": {"name": "str", "command": "str (full exec path + args)", "description": "str (optional)"},
    },
    "stop_service": {
        "fn": stop_service,
        "description": "Stop a running systemd service",
        "params": {"name": "str"},
    },
    "service_status": {
        "fn": service_status,
        "description": "Get status of running services",
        "params": {"name": "str (optional — omit to list all running services)"},
    },
    "store_cred": {
        "fn": store_cred,
        "description": "Encrypt and store a credential (API key, password, etc.) on disk",
        "params": {"name": "str", "value": "str", "description": "str (optional)"},
    },
    "get_cred": {
        "fn": get_cred,
        "description": "Retrieve a stored credential by name",
        "params": {"name": "str"},
    },
    "remember": {
        "fn": remember,
        "description": "Store a persistent fact that survives across conversations",
        "params": {"key": "str", "value": "str"},
    },
    "recall": {
        "fn": recall,
        "description": "Retrieve one or all stored memory facts",
        "params": {"key": "str (optional — omit for all memories)"},
    },
}


def execute_tool(name: str, args: dict) -> str:
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_REGISTRY[name]["fn"](**args)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_tools_description() -> str:
    lines = []
    for name, info in TOOL_REGISTRY.items():
        params_str = ", ".join(f"{k}: {v}" for k, v in info["params"].items())
        lines.append(f"- {name}({params_str})\n  → {info['description']}")
    return "\n".join(lines)
