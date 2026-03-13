"""
SynthClaw-CoAgent — Tool Implementations
12 tools the agent can invoke: shell, files, HTTP, services, credentials, memory.
"""
import json
import subprocess
import datetime
import urllib.parse
import requests
from pathlib import Path
from memory import (
    store_credential, get_credential, list_credentials,
    set_memory, get_memory, get_all_memory,
)
from config import WORKSPACE_DIR, MEDIA_DIR


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


# ── Batch tools (efficiency) ─────────────────────────────────────────────────

def read_files(paths: list) -> dict:
    """Read multiple files at once. More efficient than calling read_file many times."""
    results = {}
    for path in paths:
        p_str = str(path)
        try:
            p = Path(path) if p_str.startswith("/") else WORKSPACE_DIR / path
            if not p.exists():
                results[p_str] = {"error": f"Not found: {path}"}
            else:
                text = p.read_text()
                results[p_str] = {"content": text[:5000], "truncated": len(text) > 5000, "size": len(text)}
        except Exception as e:
            results[p_str] = {"error": str(e)}
    return {"files": results, "count": len(results)}


def write_files(files: list) -> dict:
    """Write multiple files at once. Each item: {"path": "...", "content": "..."}."""
    results = {}
    for f_item in files:
        path = f_item.get("path", "")
        content = f_item.get("content", "")
        try:
            p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            results[str(p)] = {"success": True}
        except Exception as e:
            results[str(p)] = {"error": str(e)}
    return {"files": results, "count": len(results)}


def run_commands(commands: list, timeout: int = 30) -> dict:
    """Run multiple shell commands sequentially. Returns results for each."""
    results = []
    for cmd in commands:
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(WORKSPACE_DIR),
            )
            results.append({
                "command": cmd,
                "stdout": r.stdout[-3000:],
                "stderr": r.stderr[-1000:],
                "returncode": r.returncode,
            })
        except subprocess.TimeoutExpired:
            results.append({"command": cmd, "error": f"Timed out after {timeout}s", "returncode": -1})
        except Exception as e:
            results.append({"command": cmd, "error": str(e), "returncode": -1})
    return {"results": results, "count": len(results)}


# ── Media tools ───────────────────────────────────────────────────────────────

def send_media(path: str, media_type: str = "auto", caption: str = "") -> dict:
    """Queue a file to be sent to the user via Telegram.
    The file will be sent after your text reply.
    media_type: auto, photo, video, audio, voice, document.
    """
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"File not found: {path}"}

        if media_type == "auto":
            ext = p.suffix.lower()
            if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
                media_type = "photo"
            elif ext in (".mp4", ".avi", ".mkv", ".mov", ".webm"):
                media_type = "video"
            elif ext in (".mp3", ".wav", ".flac", ".m4a", ".aac"):
                media_type = "audio"
            elif ext in (".ogg", ".opus", ".oga"):
                media_type = "voice"
            else:
                media_type = "document"

        return {
            "queued": True,
            "path": str(p),
            "type": media_type,
            "caption": caption[:1024],
            "size": p.stat().st_size,
        }
    except Exception as e:
        return {"error": str(e)}


def list_media(subfolder: str = "") -> dict:
    """List media files stored on the server.
    Optional subfolder: photos, voice, audio, video, documents, stickers, generated, downloads.
    """
    try:
        media_path = MEDIA_DIR / subfolder if subfolder else MEDIA_DIR
        if not media_path.exists():
            return {"items": [], "path": str(media_path)}

        items = []
        for f in sorted(media_path.rglob("*")):
            if f.is_file():
                items.append({
                    "name": str(f.relative_to(MEDIA_DIR)),
                    "size": f.stat().st_size,
                    "modified": datetime.datetime.fromtimestamp(
                        f.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M"),
                })
        return {"items": items[:100], "total": len(items), "path": str(media_path)}
    except Exception as e:
        return {"error": str(e)}


def download_url(url: str, filename: str = "", subfolder: str = "downloads") -> dict:
    """Download a file from a URL and save to media storage."""
    try:
        save_dir = MEDIA_DIR / subfolder
        save_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            filename = urllib.parse.urlparse(url).path.split("/")[-1] or "download"

        save_path = save_dir / filename

        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        size = save_path.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1048576 else f"{size/1048576:.1f}MB"
        return {"success": True, "path": str(save_path), "size": size_str}
    except Exception as e:
        return {"error": str(e)}


def generate_image(prompt: str, size: str = "1024x1024", filename: str = "") -> dict:
    """Generate an image using an AI model API.
    Saves to media/generated/ and queues it for sending.
    Requires an API provider that supports image generation.
    """
    try:
        import config as cfg
        from openai import OpenAI

        gen_client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_API_BASE)
        response = gen_client.images.generate(prompt=prompt, size=size, n=1)
        image_url = response.data[0].url

        save_dir = MEDIA_DIR / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{ts}.png"

        save_path = save_dir / filename
        img_resp = requests.get(image_url, timeout=120)
        img_resp.raise_for_status()
        save_path.write_bytes(img_resp.content)

        return {
            "queued": True,
            "path": str(save_path),
            "type": "photo",
            "caption": prompt[:200],
            "size": save_path.stat().st_size,
            "prompt": prompt,
        }
    except Exception as e:
        return {
            "error": f"Image generation failed: {e}",
            "hint": "Your API provider may not support image generation. "
                    "Try switching to an OpenAI-compatible endpoint that does.",
        }


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
    "read_files": {
        "fn": read_files,
        "description": "Read multiple files at once (batch). More efficient than multiple read_file calls.",
        "params": {"paths": "list[str] — list of file paths to read at once"},
    },
    "write_files": {
        "fn": write_files,
        "description": "Write multiple files at once (batch). Each item: {path, content}.",
        "params": {"files": "list[{path: str, content: str}] — list of files to write"},
    },
    "run_commands": {
        "fn": run_commands,
        "description": "Run multiple shell commands sequentially (batch). Returns all results at once.",
        "params": {
            "commands": "list[str] — list of shell commands to run",
            "timeout": "int (optional, per-command timeout, default 30)",
        },
    },
    "send_media": {
        "fn": send_media,
        "description": "Send a file from the server to the user (photo, video, audio, document). File is sent after your text reply.",
        "params": {
            "path": "str (absolute or relative to workspace)",
            "media_type": "auto/photo/video/audio/voice/document (default: auto-detect by extension)",
            "caption": "str (optional, max 1024 chars)",
        },
    },
    "list_media": {
        "fn": list_media,
        "description": "List media files stored on the server. Subfolders: photos, voice, audio, video, documents, stickers, generated, downloads",
        "params": {"subfolder": "str (optional — omit to list all media)"},
    },
    "download_url": {
        "fn": download_url,
        "description": "Download a file from a URL and save to media storage on the server",
        "params": {
            "url": "str (the URL to download)",
            "filename": "str (optional — auto-detected from URL if omitted)",
            "subfolder": "str (optional, default 'downloads')",
        },
    },
    "generate_image": {
        "fn": generate_image,
        "description": "Generate an image using an AI image model. Saves to media/generated/ and sends it to the user.",
        "params": {
            "prompt": "str (description of the image to generate)",
            "size": "str (optional: 1024x1024, 512x512, 256x256)",
            "filename": "str (optional)",
        },
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
