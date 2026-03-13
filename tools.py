"""
SynthClaw-CoAgent — Tool Implementations
12 tools the agent can invoke: shell, files, HTTP, services, credentials, memory.
"""
import json
import logging
import os
import re
import subprocess
import datetime
import urllib.parse
import requests
from pathlib import Path
from memory import (
    store_credential, get_credential, list_credentials,
    set_memory, get_memory, get_all_memory,
)
from config import (
    WORKSPACE_DIR, MEDIA_DIR,
    DEFAULT_CMD_TIMEOUT, INSTALL_CMD_TIMEOUT, BUILD_CMD_TIMEOUT,
)

logger = logging.getLogger(__name__)

# ── Timeout intelligence ─────────────────────────────────────────────────────

_INSTALL_PATTERNS = re.compile(
    r"\b(pip3?\s+install|npm\s+install|npm\s+i\b|yarn\s+add|yarn\s+install|"
    r"apt-get\s+install|apt\s+install|cargo\s+install|gem\s+install|"
    r"go\s+install|go\s+get|composer\s+install|composer\s+require)",
    re.IGNORECASE,
)

_BUILD_PATTERNS = re.compile(
    r"\b(make\b|cmake\b|cargo\s+build|go\s+build|docker\s+build|"
    r"gradle\b|mvn\b|npm\s+run\s+build|yarn\s+build|webpack\b|"
    r"gcc\b|g\+\+\b|javac\b|rustc\b|dotnet\s+build|pip3?\s+wheel)",
    re.IGNORECASE,
)

_TEST_PATTERNS = re.compile(
    r"\b(pytest|python3?\s+-m\s+(unittest|pytest)|npm\s+test|npm\s+run\s+test|"
    r"yarn\s+test|cargo\s+test|go\s+test|mvn\s+test|jest\b)",
    re.IGNORECASE,
)


def _smart_timeout(command: str, explicit_timeout: int | None) -> int:
    """Pick the right timeout based on command type.
    If the caller passed an explicit timeout, respect it.
    Otherwise, auto-detect from command content.
    """
    if explicit_timeout is not None:
        return explicit_timeout
    if _BUILD_PATTERNS.search(command):
        return BUILD_CMD_TIMEOUT
    if _INSTALL_PATTERNS.search(command) or _TEST_PATTERNS.search(command):
        return INSTALL_CMD_TIMEOUT
    return DEFAULT_CMD_TIMEOUT


def _smart_truncate(text: str, limit: int = 3000) -> str:
    """Truncate long output smartly: keep first and last lines + error lines."""
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    # Always keep error/warning lines
    error_lines = [
        l for l in lines
        if re.search(r"(error|Error|ERROR|failed|FAILED|exception|Exception|traceback|Traceback)", l)
    ]
    head = "\n".join(lines[:30])
    tail = "\n".join(lines[-20:])
    err_section = ""
    if error_lines:
        err_section = "\n... [KEY ERRORS] ...\n" + "\n".join(error_lines[:20])
    truncated = f"{head}\n\n... [{len(lines)} total lines, truncated] ...{err_section}\n\n{tail}"
    return truncated[:limit]


# ── Tool implementations ─────────────────────────────────────────────────────

def run_command(command: str, timeout: int = None) -> dict:
    """Execute a shell command. Timeout auto-detected for installs/builds (180s/300s)
    or defaults to 30s. Pass explicit timeout to override."""
    effective_timeout = _smart_timeout(command, timeout)
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=effective_timeout, cwd=str(WORKSPACE_DIR),
        )
        result = {
            "stdout": _smart_truncate(r.stdout),
            "stderr": _smart_truncate(r.stderr, 1500),
            "returncode": r.returncode,
        }
        # Auto-verify installs: if it failed, add helpful hints
        if r.returncode != 0:
            result["note"] = "⚠️ Command FAILED (non-zero exit code). Do NOT proceed as if it succeeded."
            stderr_lower = r.stderr.lower()
            if "no matching distribution" in stderr_lower or "no such package" in stderr_lower:
                result["hint"] = "Package may not exist or name is wrong. Check the exact package name."
            elif "permission denied" in stderr_lower:
                result["hint"] = "Try with sudo or check file permissions."
            elif "already in use" in stderr_lower or "address already in use" in stderr_lower:
                result["hint"] = "Port is in use. Find the process: lsof -i :<port> or kill it."
            elif "modulenotfounderror" in stderr_lower or "no module named" in stderr_lower:
                pkg_match = re.search(r"no module named ['\'\"\']?(\w+)", stderr_lower)
                if pkg_match:
                    result["hint"] = f"Missing module '{pkg_match.group(1)}'. Install it first."
        return result
    except subprocess.TimeoutExpired:
        # Auto-retry ONCE with 2x timeout for install/build commands
        if _INSTALL_PATTERNS.search(command) or _BUILD_PATTERNS.search(command):
            retry_timeout = effective_timeout * 2
            logger.warning(f"Command timed out at {effective_timeout}s, retrying with {retry_timeout}s: {command[:100]}")
            try:
                r = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    timeout=retry_timeout, cwd=str(WORKSPACE_DIR),
                )
                result = {
                    "stdout": _smart_truncate(r.stdout),
                    "stderr": _smart_truncate(r.stderr, 1500),
                    "returncode": r.returncode,
                    "note": f"Completed on retry (took >{effective_timeout}s, retried with {retry_timeout}s timeout).",
                }
                if r.returncode != 0:
                    result["note"] += " ⚠️ Command FAILED (non-zero exit code)."
                return result
            except subprocess.TimeoutExpired:
                return {
                    "error": f"Timed out after {effective_timeout}s + retry at {retry_timeout}s. Command is too slow.",
                    "returncode": -1,
                    "hint": "Try breaking this into smaller steps, or run it as a background service with spawn_service.",
                }
        return {"error": f"Timed out after {effective_timeout}s", "returncode": -1}
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


def run_commands(commands: list, timeout: int = None) -> dict:
    """Run multiple shell commands sequentially. Timeout auto-detected per command."""
    results = []
    for cmd in commands:
        cmd_timeout = _smart_timeout(cmd, timeout)
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=cmd_timeout, cwd=str(WORKSPACE_DIR),
            )
            entry = {
                "command": cmd,
                "stdout": _smart_truncate(r.stdout),
                "stderr": _smart_truncate(r.stderr, 1500),
                "returncode": r.returncode,
            }
            if r.returncode != 0:
                entry["note"] = "⚠️ FAILED"
            results.append(entry)
        except subprocess.TimeoutExpired:
            results.append({"command": cmd, "error": f"Timed out after {cmd_timeout}s", "returncode": -1})
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


# ── Utility tools ─────────────────────────────────────────────────────────────

def search_files(pattern: str, path: str = ".", max_results: int = 50) -> dict:
    """Search for text/regex pattern across files in a directory (recursive grep)."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"Path not found: {path}"}
        r = subprocess.run(
            f"grep -rnI --include='*' '{pattern}' '{p}'",
            shell=True, capture_output=True, text=True,
            timeout=30, cwd=str(WORKSPACE_DIR),
        )
        lines = r.stdout.strip().splitlines()
        return {
            "matches": lines[:max_results],
            "total": len(lines),
            "truncated": len(lines) > max_results,
        }
    except subprocess.TimeoutExpired:
        return {"error": "Search timed out after 30s. Try a narrower path or simpler pattern."}
    except Exception as e:
        return {"error": str(e)}


def patch_file(path: str, old_text: str, new_text: str) -> dict:
    """Replace a specific text snippet in a file. More precise than rewriting the whole file.
    The old_text must match EXACTLY (including whitespace)."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"File not found: {path}"}
        content = p.read_text()
        count = content.count(old_text)
        if count == 0:
            return {"error": "old_text not found in file. Read the file first to get the exact text."}
        if count > 1:
            return {"error": f"old_text matches {count} locations. Make it more specific (include more surrounding lines)."}
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content)
        return {"success": True, "path": str(p), "replacements": 1}
    except Exception as e:
        return {"error": str(e)}


def system_info() -> dict:
    """Get system information: CPU, memory, disk, OS, uptime in one call."""
    info = {}
    try:
        # OS
        r = subprocess.run("cat /etc/os-release | head -3", shell=True, capture_output=True, text=True, timeout=5)
        info["os"] = r.stdout.strip()
        # Uptime
        r = subprocess.run("uptime -p", shell=True, capture_output=True, text=True, timeout=5)
        info["uptime"] = r.stdout.strip()
        # CPU
        r = subprocess.run("nproc", shell=True, capture_output=True, text=True, timeout=5)
        info["cpu_cores"] = r.stdout.strip()
        r = subprocess.run("cat /proc/loadavg", shell=True, capture_output=True, text=True, timeout=5)
        info["load_avg"] = r.stdout.strip()
        # Memory
        r = subprocess.run("free -h | head -2", shell=True, capture_output=True, text=True, timeout=5)
        info["memory"] = r.stdout.strip()
        # Disk
        r = subprocess.run("df -h / | tail -1", shell=True, capture_output=True, text=True, timeout=5)
        info["disk"] = r.stdout.strip()
        # Python & Node versions
        r = subprocess.run("python3 --version 2>&1; node --version 2>&1", shell=True, capture_output=True, text=True, timeout=5)
        info["runtimes"] = r.stdout.strip()
    except Exception as e:
        info["error"] = str(e)
    return info


def check_port(port: int) -> dict:
    """Check if a network port is in use. Shows what process is using it."""
    try:
        r = subprocess.run(
            f"ss -tlnp | grep :{port} || echo 'Port {port} is free'",
            shell=True, capture_output=True, text=True, timeout=10,
        )
        in_use = f":{port}" in r.stdout and "free" not in r.stdout
        return {
            "port": port,
            "in_use": in_use,
            "details": r.stdout.strip(),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "run_command": {
        "fn": run_command,
        "description": "Execute a shell command. Timeout auto-scales: 30s normal, 180s installs, 300s builds. Override with explicit timeout. Returns returncode (0=success, non-zero=FAILED).",
        "params": {"command": "str (required)", "timeout": "int (optional, auto-detected if omitted)"},
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
        "description": "Run multiple shell commands sequentially (batch). Timeout auto-detected per command. Returns all results with returncode (0=success).",
        "params": {
            "commands": "list[str] — list of shell commands to run",
            "timeout": "int (optional, per-command, auto-detected if omitted)",
        },
    },
    "search_files": {
        "fn": search_files,
        "description": "Search for a text/regex pattern across files recursively (like grep -rn). Use this instead of run_command with grep.",
        "params": {
            "pattern": "str — text or regex to search for",
            "path": "str (optional, default '.' = workspace root)",
            "max_results": "int (optional, default 50)",
        },
    },
    "patch_file": {
        "fn": patch_file,
        "description": "Replace a specific text snippet in a file (find-and-replace). More precise than rewriting the whole file. old_text must match exactly ONE location.",
        "params": {
            "path": "str",
            "old_text": "str — exact text to find (include surrounding lines for uniqueness)",
            "new_text": "str — replacement text",
        },
    },
    "system_info": {
        "fn": system_info,
        "description": "Get system info (CPU, memory, disk, OS, uptime, runtimes) in one call. Use this instead of multiple run_command calls.",
        "params": {},
    },
    "check_port": {
        "fn": check_port,
        "description": "Check if a network port is in use and what process is using it",
        "params": {"port": "int — port number to check"},
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
