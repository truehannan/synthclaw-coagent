from __future__ import annotations
import csv
import json
import logging
import os
import re
import smtplib
import socket
import sqlite3
import subprocess
import datetime
import tempfile
import time
import urllib.parse
import requests
from email.message import EmailMessage
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
    if text is None:          # Fix #3: guard against None input
        return ""
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

# Venv paths — all Python/pip operations must use these
_VENV_PYTHON = "/opt/agent/venv/bin/python"
_VENV_PIP    = "/opt/agent/venv/bin/pip"


def _fix_pip_path(command: str) -> str:
    """Rewrite bare pip/pip3/python/python3 calls to the venv binaries.
    This ensures installs always land in the agent venv, never the system site."""
    # pip3 install / pip install → venv pip (only when not already an absolute path)
    command = re.sub(r'(?<![/\w])pip3?\s+', f'{_VENV_PIP} ', command)
    # python3 / python → venv python (only bare invocations)
    command = re.sub(r'(?<![/\w])python3?\s+', f'{_VENV_PYTHON} ', command)
    return command


def run_command(command: str, timeout: int = None) -> dict:
    """Execute a shell command. Timeout auto-detected for installs/builds (180s/300s)
    or defaults to 30s. Pass explicit timeout to override.
    pip/pip3 and python/python3 are automatically redirected to the agent venv."""
    command = _fix_pip_path(command)
    effective_timeout = _smart_timeout(command, timeout)
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=effective_timeout, cwd=str(WORKSPACE_DIR),
        )
        result = {
            "stdout": _smart_truncate(r.stdout or ""),   # Fix #4: guard None stdout
            "stderr": _smart_truncate(r.stderr or "", 1500),  # Fix #4: guard None stderr
            "returncode": r.returncode,
        }
        # Auto-verify installs: if it failed, add helpful hints
        if r.returncode != 0:
            result["note"] = "⚠️ Command FAILED (non-zero exit code). Do NOT proceed as if it succeeded."
            stderr_lower = r.stderr.lower()
            if "no matching distribution" in stderr_lower or "no such package" in stderr_lower:
                result["hint"] = f"Package name may be wrong. Run `{_VENV_PIP} index versions <pkg>` to find the correct name, then retry."
            elif "permission denied" in stderr_lower:
                result["hint"] = "Permission denied. Retry prefixed with sudo, or fix ownership with chown."
            elif "already in use" in stderr_lower or "address already in use" in stderr_lower:
                result["hint"] = "Port in use. Run kill_process(port=N) or `fuser -k <port>/tcp` to free it, then retry."
            elif "modulenotfounderror" in stderr_lower or "no module named" in stderr_lower:
                pkg_match = re.search(r"no module named ['\"]?(\w+)", stderr_lower)
                if pkg_match:
                    result["hint"] = f"Module '{pkg_match.group(1)}' missing. Run `{_VENV_PIP} install {pkg_match.group(1)}` then retry."
            elif "externally-managed-environment" in stderr_lower:
                result["hint"] = f"System Python is PEP 668 locked. Use the venv: `{_VENV_PIP} install <pkg>`."
            elif "ssl" in stderr_lower or "certificate" in stderr_lower:
                result["hint"] = f"SSL error. Retry: `{_VENV_PIP} install --trusted-host pypi.org --trusted-host files.pythonhosted.org <pkg>`."
            elif "could not find a version" in stderr_lower:
                result["hint"] = f"Version not found. Run `{_VENV_PIP} index versions <pkg>` to list valid versions, then retry."
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
                    "stdout": _smart_truncate(r.stdout or ""),   # Fix #4
                    "stderr": _smart_truncate(r.stderr or "", 1500),
                    "returncode": r.returncode,
                    "note": f"Completed on retry (took >{effective_timeout}s, retried with {retry_timeout}s timeout).",
                    "timeout_retry": True,   # Fix #8: flag so LLM can handle slow cmds differently
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
        text = p.read_text(errors="replace")
        if not text:                        # Fix #9: guard empty/unreadable file
            return {"error": f"File is empty: {path}", "size": 0}
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
    """Store an encrypted credential. Auto-detects and registers known APIs."""
    try:
        store_credential(name, value, description)
        result = {"success": True, "stored": name}

        # Auto-detect: is this credential for a known API?
        from known_apis import detect_api_from_key
        from memory import register_dynamic_tool, get_dynamic_tool

        detected = detect_api_from_key(value, f"{name} {description}")
        if detected and not get_dynamic_tool(detected["name"]):
            # Auto-register the API
            register_dynamic_tool(
                name=detected["name"],
                base_url=detected["base_url"],
                auth_cred=name,  # use the credential name the user just stored
                auth_type=detected.get("auth_type", "bearer"),
                auth_header=detected.get("auth_header", "Authorization"),
                auth_prefix=detected.get("auth_prefix", "Bearer "),
                endpoints=detected.get("endpoints", []),
                description=detected.get("description", ""),
                docs_url=detected.get("docs_url", ""),
            )
            result["api_registered"] = detected["name"]
            result["api_description"] = detected.get("description", "")
            result["api_endpoints"] = len(detected.get("endpoints", []))

        return result
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
        effective_timeout = _smart_timeout(cmd, timeout)
        try:
            r = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=effective_timeout, cwd=str(WORKSPACE_DIR),
            )
            entry = {
                "command": cmd,
                "stdout": _smart_truncate(r.stdout),
                "stderr": _smart_truncate(r.stderr, 1500),
                "returncode": r.returncode,
            }
            if r.returncode != 0:
                entry["note"] = "⚠️ FAILED (non-zero exit code). Check stderr."
            results.append(entry)
        except subprocess.TimeoutExpired:
            results.append({"command": cmd, "error": f"Timed out after {effective_timeout}s", "returncode": -1})
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
        if resp.status_code >= 400:          # Fix #6
            return {"error": f"HTTP {resp.status_code} downloading {url}"}
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

        size = save_path.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1048576 else f"{size/1048576:.1f}MB"
        return {"success": True, "path": str(save_path), "size": size_str}
    except Exception as e:
        save_path.unlink(missing_ok=True)    # Fix #20: delete partial/corrupt file on failure
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


# ── Extended skills ───────────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 8) -> dict:
    """Search the web using DuckDuckGo. Returns titles, URLs, and snippets."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return {"query": query, "results": results, "count": len(results)}
    except ImportError:
        return {"error": "duckduckgo-search not installed. Run: pip install duckduckgo-search"}
    except Exception as e:
        return {"error": str(e)}


def google_search(query: str, max_results: int = 5) -> dict:
    """Search Google via Custom Search JSON API. Requires GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX
    stored as credentials or env vars. Falls back to DuckDuckGo if not configured."""
    from config import GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_CX
    api_key = get_credential("GOOGLE_SEARCH_API_KEY") or GOOGLE_SEARCH_API_KEY
    cx = get_credential("GOOGLE_SEARCH_CX") or GOOGLE_SEARCH_CX

    if not api_key or not cx:
        # Fallback to DuckDuckGo
        return web_search(query, max_results)

    try:
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(max_results, 10),
        }
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return {"query": query, "results": results, "count": len(results), "engine": "google"}
    except Exception as e:
        # Fallback to DuckDuckGo on any Google error
        fallback = web_search(query, max_results)
        fallback["note"] = f"Google failed ({e}), used DuckDuckGo fallback"
        return fallback


def scrape_page(url: str, max_chars: int = 8000) -> dict:
    """Fetch a web page and return clean readable text (scripts/styles/nav removed)."""
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        if resp.status_code >= 400:          # Fix #6: explicit HTTP error check
            return {"error": f"HTTP {resp.status_code} from {url}"}
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        cleaned = "\n".join(lines)
        truncated = len(cleaned) > max_chars
        return {
            "url": url,
            "text": cleaned[:max_chars],
            "chars": len(cleaned),
            "truncated": truncated,
        }
    except ImportError:
        return {"error": "beautifulsoup4 not installed. Run: pip install beautifulsoup4"}
    except Exception as e:
        return {"error": str(e)}


def scrape_selector(url: str, selector: str, max_results: int = 20) -> dict:
    """Extract elements matching a CSS selector from a web page."""
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        if resp.status_code >= 400:          # Fix #6
            return {"error": f"HTTP {resp.status_code} from {url}"}
        soup = BeautifulSoup(resp.text, "html.parser")
        elements = soup.select(selector)[:max_results]
        results = [el.get_text(strip=True) for el in elements]
        return {"url": url, "selector": selector, "matches": results, "count": len(results)}
    except ImportError:
        return {"error": "beautifulsoup4 not installed. Run: pip install beautifulsoup4"}
    except Exception as e:
        return {"error": str(e)}


def run_python(code: str, timeout: int = 30) -> dict:
    """Execute a Python code snippet in a subprocess. Returns stdout, stderr, returncode."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "snippet.py"
            script.write_text(code)
            r = subprocess.run(
                ["/opt/agent/venv/bin/python", str(script)],
                capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
            )
            return {
                "stdout": r.stdout[:4000],
                "stderr": r.stderr[:2000],
                "returncode": r.returncode,
            }
    except subprocess.TimeoutExpired:
        return {"error": f"Code timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def exec_code(code: str, lang: str = "node", timeout: int = 30) -> dict:
    """Execute code directly in memory — no files written to disk.
    lang: 'node' (JavaScript/TypeScript), 'python', or 'bash'.
    This is the PREFERRED way to run quick computations, data processing,
    API calls, JSON parsing, or any logic that doesn't need persistence.
    Use this instead of write_file + run_command for one-off code execution."""
    try:
        if lang in ("node", "js", "javascript"):
            r = subprocess.run(
                ["node", "-e", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(WORKSPACE_DIR),
            )
        elif lang in ("python", "py"):
            # Use venv python if available, else system python
            py_bin = str(WORKSPACE_DIR.parent / "venv" / "bin" / "python")
            if not Path(py_bin).exists():
                py_bin = "python3"
            r = subprocess.run(
                [py_bin, "-c", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(WORKSPACE_DIR),
            )
        elif lang in ("bash", "sh"):
            r = subprocess.run(
                ["bash", "-c", code],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(WORKSPACE_DIR),
            )
        else:
            return {"error": f"Unsupported language: {lang}. Use: node, python, or bash."}

        return {
            "stdout": r.stdout[:4000] if r.stdout else "",
            "stderr": r.stderr[:2000] if r.stderr else "",
            "returncode": r.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Code timed out after {timeout}s"}
    except FileNotFoundError as e:
        return {"error": f"Runtime not found: {e}. Install node/python first."}
    except Exception as e:
        return {"error": str(e)}


def read_csv(path: str, max_rows: int = 100) -> dict:
    """Read a CSV file and return its contents as a list of dicts."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"File not found: {path}"}
        with open(p, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for _, row in zip(range(max_rows), reader)]
        return {"path": str(p), "rows": rows, "count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


def query_csv(path: str, filter_col: str, filter_val: str, max_rows: int = 100) -> dict:
    """Filter rows from a CSV where filter_col equals filter_val."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"File not found: {path}"}
        matched = []
        with open(p, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get(filter_col, "") == filter_val:
                    matched.append(dict(row))
                    if len(matched) >= max_rows:
                        break
        return {"matches": matched, "count": len(matched)}
    except Exception as e:
        return {"error": str(e)}


def summarize_file(path: str, max_chars: int = 4000) -> dict:
    """Read a large file and return a sampled view: head, middle chunk, tail."""
    try:
        p = Path(path) if path.startswith("/") else WORKSPACE_DIR / path
        if not p.exists():
            return {"error": f"File not found: {path}"}
        content = p.read_text(errors="replace")
        total = len(content)
        if total <= max_chars:
            return {"path": str(p), "total_chars": total, "content": content, "sampled": False}
        head = content[:2000]
        mid_start = total // 2 - 500
        mid = content[mid_start:mid_start + 1000]
        tail = content[-1000:]
        return {
            "path": str(p),
            "total_chars": total,
            "sampled": True,
            "head": head,
            "middle": mid,
            "tail": tail,
            "hint": "File too large to show fully. Use read_file for specific sections.",
        }
    except Exception as e:
        return {"error": str(e)}


def think(thought: str) -> dict:
    """Reasoning scratchpad — write out your thinking. Output is NOT shown to the user.
    Use this to plan multi-step tasks, reason through trade-offs, or avoid mistakes."""
    return {"thought_recorded": True, "length": len(thought)}


def ask_user(question: str) -> dict:
    """Pause the task and ask the user a clarifying question before continuing.
    The question will be sent to the user, and their reply will come in the next message."""
    return {"ask_user": True, "question": question}


def send_email(to: str, subject: str, body: str, from_addr: str = "", smtp_cred: str = "smtp") -> dict:
    """Send an email via SMTP. Credentials must be stored with store_cred as 'smtp'
    in format 'host:port:user:password' (e.g. smtp.gmail.com:587:you@gmail.com:apppassword)."""
    try:
        cred_val = get_credential(smtp_cred)
        if not cred_val:
            return {"error": f"No SMTP credential stored under '{smtp_cred}'. Use store_cred to save 'host:port:user:password'."}
        parts = cred_val.split(":", 3)
        if len(parts) != 4:
            return {"error": "SMTP credential must be 'host:port:user:password'"}
        host, port_str, user, password = parts
        port = int(port_str)
        sender = from_addr or user
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        msg.set_content(body)
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        return {"success": True, "to": to, "subject": subject}
    except Exception as e:
        return {"error": str(e)}


def send_telegram_message(chat_id: str, text: str) -> dict:
    """Send a Telegram message to any chat ID using the bot token.
    Useful for notifications, alerts, or messaging other chats."""
    try:
        from config import TELEGRAM_BOT_TOKEN
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=15)
        data = resp.json()
        if data.get("ok"):
            return {"success": True, "message_id": data["result"]["message_id"]}
        return {"error": data.get("description", "Unknown Telegram error")}
    except Exception as e:
        return {"error": str(e)}


def set_reminder(message: str, schedule: str, name: str = "") -> dict:
    """Set a timed reminder that sends a Telegram message to the owner at the specified time.
    schedule: cron expression (e.g. '30 9 * * *' = daily 9:30am, '0 14 * * 1' = Monday 2pm).
    The agent will message the owner at that time. Use for: reminders, daily reports, recurring checks."""
    try:
        from config import TELEGRAM_TOKEN, BASE_DIR
        from memory import get_config
        owner_id = get_config("owner_telegram_id")
        if not owner_id:
            return {"error": "No owner registered yet. The owner must /start the bot first."}

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name or f"reminder_{int(time.time())}")
        # Escape message for shell
        escaped_msg = message.replace("'", "'\\''")
        # Build curl command that hits Telegram API directly
        cmd = (
            f"curl -s -X POST 'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage' "
            f"-d chat_id={owner_id} -d text='{escaped_msg}'"
        )
        cron_line = f"{schedule} {cmd} # agent-task:{safe_name}"

        r = subprocess.run("crontab -l 2>/dev/null", shell=True, capture_output=True, text=True)
        existing = r.stdout.strip()
        if f"agent-task:{safe_name}" in existing:
            return {"error": f"Reminder '{safe_name}' already exists. Use remove_cron to remove it first."}
        new_crontab = (existing + "\n" + cron_line).strip() + "\n"
        r2 = subprocess.run("crontab -", input=new_crontab, shell=True, capture_output=True, text=True)
        if r2.returncode != 0:
            return {"error": r2.stderr.strip() or "crontab write failed"}
        return {
            "success": True,
            "name": safe_name,
            "schedule": schedule,
            "message": message,
            "note": f"Reminder set. Will message owner at: {schedule}",
        }
    except Exception as e:
        return {"error": str(e)}


def schedule_task(name: str, command: str, schedule: str) -> dict:
    """Create a cron job. schedule is a standard cron expression (e.g. '0 * * * *' = every hour).
    name is used as a comment tag to identify the job later."""
    try:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        cron_line = f"{schedule} {command} # agent-task:{safe_name}"
        r = subprocess.run("crontab -l 2>/dev/null", shell=True, capture_output=True, text=True)
        existing = r.stdout.strip()
        if f"agent-task:{safe_name}" in existing:
            return {"error": f"A cron job named '{safe_name}' already exists. Use remove_cron first."}
        new_crontab = (existing + "\n" + cron_line).strip() + "\n"
        r2 = subprocess.run("crontab -", input=new_crontab, shell=True, capture_output=True, text=True)
        if r2.returncode != 0:
            return {"error": r2.stderr.strip() or "crontab write failed"}
        return {"success": True, "name": safe_name, "schedule": schedule, "command": command}
    except Exception as e:
        return {"error": str(e)}


def list_cron() -> dict:
    """List all cron jobs, highlighting ones created by the agent."""
    try:
        r = subprocess.run("crontab -l 2>/dev/null", shell=True, capture_output=True, text=True)
        all_jobs = r.stdout.strip().splitlines()
        agent_jobs = [l for l in all_jobs if "agent-task:" in l]
        return {
            "all_jobs": all_jobs,
            "agent_jobs": agent_jobs,
            "total": len(all_jobs),
        }
    except Exception as e:
        return {"error": str(e)}


def remove_cron(name: str) -> dict:
    """Remove a cron job created by the agent (by its name tag)."""
    try:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        r = subprocess.run("crontab -l 2>/dev/null", shell=True, capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        new_lines = [l for l in lines if f"agent-task:{safe_name}" not in l]
        if len(new_lines) == len(lines):
            return {"error": f"No cron job named '{safe_name}' found"}
        new_crontab = "\n".join(new_lines).strip() + "\n"
        r2 = subprocess.run("crontab -", input=new_crontab, shell=True, capture_output=True, text=True)
        if r2.returncode != 0:
            return {"error": r2.stderr.strip() or "crontab write failed"}
        return {"success": True, "removed": safe_name}
    except Exception as e:
        return {"error": str(e)}


def tail_logs(service: str = "", path: str = "", lines: int = 50) -> dict:
    """Tail log lines from a systemd service or a log file path."""
    try:
        if service:
            r = subprocess.run(
                f"journalctl -u {service} -n {lines} --no-pager 2>&1",
                shell=True, capture_output=True, text=True, timeout=15,
            )
        elif path:
            lp = Path(path)
            if not lp.exists():
                return {"error": f"Log file not found: {path}"}
            r = subprocess.run(
                f"tail -n {lines} '{path}'",
                shell=True, capture_output=True, text=True, timeout=15,
            )
        else:
            return {"error": "Provide either service name or log file path"}
        return {"output": r.stdout.strip(), "lines_requested": lines}
    except subprocess.TimeoutExpired:
        return {"error": "Log tail timed out"}
    except Exception as e:
        return {"error": str(e)}


def watch_url(url: str, timeout: int = 60, interval: int = 5) -> dict:
    """Poll a URL until it returns HTTP 200 or timeout is reached. Returns final status."""
    start = time.time()
    attempts = 0
    last_status = None
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=10)
            last_status = resp.status_code
            if resp.status_code == 200:
                return {"success": True, "url": url, "attempts": attempts + 1, "elapsed": round(time.time() - start, 1)}
        except Exception as ex:
            last_status = str(ex)
        attempts += 1
        time.sleep(interval)
    return {"success": False, "url": url, "attempts": attempts, "last_status": last_status, "elapsed": timeout}


def kill_process(name: str = "", port: int = 0) -> dict:
    """Kill a process by name (pkill) or by port number (fuser -k)."""
    try:
        if port:
            r = subprocess.run(f"fuser -k {port}/tcp 2>&1", shell=True, capture_output=True, text=True, timeout=10)
            return {"port": port, "output": r.stdout.strip() or "Done", "returncode": r.returncode}
        if name:
            r = subprocess.run(f"pkill -f '{name}' 2>&1", shell=True, capture_output=True, text=True, timeout=10)
            return {"name": name, "killed": r.returncode == 0, "returncode": r.returncode, "output": r.stdout.strip()}
        return {"error": "Provide either name or port"}
    except Exception as e:
        return {"error": str(e)}


def sqlite_query(db_path: str, query: str, params: list = None) -> dict:
    """Run a SQL query (SELECT or DML) against a SQLite .db file. Returns rows as list of dicts."""
    try:
        p = Path(db_path) if db_path.startswith("/") else WORKSPACE_DIR / db_path
        conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, params or [])
        if query.strip().upper().startswith("SELECT"):
            rows = [dict(row) for row in cur.fetchmany(200)]
            conn.close()
            return {"rows": rows, "count": len(rows)}
        else:
            conn.commit()
            affected = cur.rowcount
            conn.close()
            return {"success": True, "rows_affected": affected}
    except Exception as e:
        return {"error": str(e)}


def redis_get(key: str, host: str = "127.0.0.1", port: int = 6379) -> dict:
    """Get a value from Redis by key."""
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=host, port=port, decode_responses=True)
        val = r.get(key)
        return {"key": key, "value": val, "exists": val is not None}
    except ImportError:
        return {"error": "redis not installed. Run: pip install redis"}
    except Exception as e:
        return {"error": str(e)}


def redis_set(key: str, value: str, ttl: int = 0, host: str = "127.0.0.1", port: int = 6379) -> dict:
    """Set a value in Redis. Optionally set TTL in seconds (0 = no expiry)."""
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=host, port=port, decode_responses=True)
        if ttl > 0:
            r.setex(key, ttl, value)
        else:
            r.set(key, value)
        return {"success": True, "key": key, "ttl": ttl or "none"}
    except ImportError:
        return {"error": "redis not installed. Run: pip install redis"}
    except Exception as e:
        return {"error": str(e)}


# ── Dynamic API tools ─────────────────────────────────────────────────────────

def register_api(name: str, base_url: str, auth_cred: str, auth_type: str = "bearer",
                 auth_header: str = "Authorization", auth_prefix: str = "Bearer ",
                 endpoints: str = "[]", description: str = "", docs_url: str = "") -> dict:
    """Register an external API as a callable tool. After registration, the agent can call it via api_call.
    auth_type: bearer, header, query, basic, body.
    endpoints: JSON string of [{method, path, desc}] — key operations this API supports.
    """
    from memory import register_dynamic_tool
    from known_apis import detect_api_from_key, KNOWN_APIS

    name = name.lower().strip().replace(" ", "_")

    # Try to auto-detect from known APIs if minimal info given
    if not base_url or base_url == "auto":
        detected = None
        # Check by name
        if name in KNOWN_APIS:
            detected = KNOWN_APIS[name]
        # Check by credential prefix
        if not detected:
            cred_val = get_credential(auth_cred) or ""
            detected = detect_api_from_key(cred_val, name)
        if detected:
            base_url = detected["base_url"]
            auth_type = detected.get("auth_type", auth_type)
            auth_header = detected.get("auth_header", auth_header)
            auth_prefix = detected.get("auth_prefix", auth_prefix)
            description = detected.get("description", description)
            docs_url = detected.get("docs_url", docs_url)
            endpoints = json.dumps(detected.get("endpoints", []))

    if not base_url:
        return {"error": "Could not determine base_url. Provide it explicitly or use a known API name."}

    # Parse endpoints
    try:
        ep_list = json.loads(endpoints) if isinstance(endpoints, str) else endpoints
    except json.JSONDecodeError:
        ep_list = []

    register_dynamic_tool(
        name=name,
        base_url=base_url,
        auth_cred=auth_cred,
        auth_type=auth_type,
        auth_header=auth_header,
        auth_prefix=auth_prefix,
        endpoints=ep_list,
        description=description,
        docs_url=docs_url,
    )
    return {
        "success": True,
        "name": name,
        "base_url": base_url,
        "auth_type": auth_type,
        "endpoints_count": len(ep_list),
        "description": description,
    }


def api_call(api: str, method: str = "GET", path: str = "/", body: str = "",
             query_params: str = "", headers: str = "") -> dict:
    """Call a registered API. The agent uses this to hit any previously registered external API.
    api: name of the registered API (e.g. 'stripe', 'vercel', 'cloudflare')
    method: HTTP method (GET, POST, PUT, DELETE, PATCH)
    path: endpoint path (e.g. '/charges', '/v9/projects')
    body: JSON body for POST/PUT/PATCH (string)
    query_params: URL query string (e.g. 'limit=10&status=active')
    headers: additional headers as JSON object string (optional)
    """
    from memory import get_dynamic_tool, get_credential

    tool = get_dynamic_tool(api)
    if not tool:
        return {"error": f"API '{api}' not registered. Use register_api first or provide the credential."}
    if not tool.get("enabled"):
        return {"error": f"API '{api}' is disabled."}

    # Get auth credential
    cred_value = get_credential(tool["auth_cred"])
    if not cred_value:
        return {"error": f"Credential '{tool['auth_cred']}' not found. Store it with store_cred."}

    # Build URL
    base = tool["base_url"].rstrip("/")
    endpoint = path if path.startswith("/") or path.startswith("?") else f"/{path}"
    url = f"{base}{endpoint}"
    if query_params:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{query_params}"

    # Build auth headers
    req_headers = {"Content-Type": "application/json"}
    auth_type = tool.get("auth_type", "bearer")

    if auth_type == "bearer":
        prefix = tool.get("auth_prefix", "Bearer ")
        req_headers[tool.get("auth_header", "Authorization")] = f"{prefix}{cred_value}"
    elif auth_type == "header":
        req_headers[tool.get("auth_header", "X-API-Key")] = cred_value
    elif auth_type == "basic":
        import base64
        encoded = base64.b64encode(cred_value.encode()).decode()
        req_headers["Authorization"] = f"Basic {encoded}"
    elif auth_type == "query":
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}key={cred_value}"
    elif auth_type == "body":
        # Auth goes in POST body (e.g. UptimeRobot)
        pass

    # Merge extra headers
    if headers:
        try:
            extra = json.loads(headers)
            req_headers.update(extra)
        except json.JSONDecodeError:
            pass

    # Build body
    json_body = None
    if body:
        try:
            json_body = json.loads(body)
        except json.JSONDecodeError:
            json_body = None

    # For body-auth APIs, inject key into body
    if auth_type == "body" and json_body is None:
        json_body = {}
    if auth_type == "body" and isinstance(json_body, dict):
        json_body["api_key"] = cred_value

    # Make the request
    try:
        resp = requests.request(
            method=method.upper(),
            url=url,
            json=json_body if json_body else None,
            data=body if (body and not json_body and method.upper() in ("POST", "PUT", "PATCH")) else None,
            headers=req_headers,
            timeout=30,
        )

        # Parse response
        try:
            resp_data = resp.json()
        except Exception:
            resp_data = resp.text[:3000]

        result = {
            "status": resp.status_code,
            "ok": 200 <= resp.status_code < 300,
            "api": api,
            "method": method.upper(),
            "path": path,
        }

        if isinstance(resp_data, dict):
            # Truncate large responses
            resp_str = json.dumps(resp_data)
            if len(resp_str) > 4000:
                result["data"] = json.dumps(resp_data, indent=2)[:3500] + "\n... [truncated]"
            else:
                result["data"] = resp_data
        elif isinstance(resp_data, str):
            result["data"] = resp_data[:3000]
        else:
            result["data"] = resp_data

        return result
    except requests.Timeout:
        return {"error": f"Request to {api} timed out (30s)", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


def list_apis() -> dict:
    """List all registered dynamic APIs and their capabilities."""
    from memory import list_dynamic_tools
    tools = list_dynamic_tools()
    if not tools:
        return {"apis": [], "count": 0, "note": "No APIs registered. Use register_api or give me a credential."}
    return {"apis": tools, "count": len(tools)}


def composio_execute(tool_slug: str, args: str = "{}", connected_account_id: str = "") -> dict:
    """Execute a Composio tool action. Composio provides 1000+ app integrations (Gmail, Slack, GitHub, etc.).
    tool_slug: the Composio tool slug (e.g. 'GMAIL_SEND_EMAIL', 'GITHUB_CREATE_ISSUE')
    args: JSON string of arguments for the tool
    connected_account_id: optional, Composio connected account ID"""
    from config import COMPOSIO_API_KEY
    from memory import get_credential
    api_key = get_credential("COMPOSIO_API_KEY") or COMPOSIO_API_KEY
    if not api_key:
        return {"error": "Composio not configured. Store COMPOSIO_API_KEY first."}
    try:
        body = {"arguments": json.loads(args) if isinstance(args, str) else args}
        if connected_account_id:
            body["connectedAccountId"] = connected_account_id
        headers = {"x-api-key": api_key, "Content-Type": "application/json"}
        resp = requests.post(
            f"https://backend.composio.dev/api/v3.1/tools/execute/{tool_slug}",
            headers=headers, json=body, timeout=30,
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            return {"success": True, "tool": tool_slug, "result": data}
        return {"error": f"Composio returned {resp.status_code}", "details": resp.text[:500]}
    except Exception as e:
        return {"error": str(e)}


# ── Skill management tools ─────────────────────────────────────────────────────

def install_skill(source: str, name: str = "") -> dict:
    """Install a skill from a source URI.
    source formats:
      - URL to .zip: https://example.com/skill.zip
      - GitHub repo: github:owner/repo
      - ClawHub: clawhub:@owner/skill-name
    The source is recorded in skill_sources (D1/local) for auto-reinstall.
    """
    import zipfile
    import io
    import shutil
    from memory import add_skill_source, list_skill_sources

    SKILLS_DIR = Path(os.getenv("SYNTHCLAW_BASE_DIR", "/opt/agent")) / ".skills"
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    source = source.strip()
    source_type = "url"
    source_uri = source

    try:
        if source.startswith("clawhub:"):
            # ClawHub format: clawhub:@owner/name
            source_type = "clawhub"
            source_uri = source
            # ClawHub resolves to GitHub for now
            ref = source.replace("clawhub:", "").lstrip("@")
            zip_url = f"https://github.com/{ref}/archive/refs/heads/main.zip"
            skill_name = name or ref.split("/")[-1]
        elif source.startswith("github:"):
            source_type = "github"
            source_uri = source
            ref = source.replace("github:", "")
            zip_url = f"https://github.com/{ref}/archive/refs/heads/main.zip"
            skill_name = name or ref.split("/")[-1]
        elif source.startswith("http"):
            zip_url = source
            skill_name = name or source.split("/")[-1].replace(".zip", "")
        else:
            # Assume it's a GitHub search term
            source_type = "github"
            search_url = f"https://api.github.com/search/repositories?q={source}+skill+in:name&sort=stars&per_page=1"
            resp = requests.get(search_url, timeout=15)
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    repo = items[0]
                    source_uri = f"github:{repo['full_name']}"
                    zip_url = f"https://github.com/{repo['full_name']}/archive/refs/heads/{repo.get('default_branch', 'main')}.zip"
                    skill_name = name or repo["name"]
                else:
                    return {"error": f"No skill found matching: {source}"}
            else:
                return {"error": f"GitHub search failed: {resp.status_code}"}

        # Download and extract
        resp = requests.get(zip_url, timeout=60)
        if resp.status_code != 200:
            return {"error": f"Failed to download: HTTP {resp.status_code}"}

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # Extract to a temp dir, then move to skills dir
            names = zf.namelist()
            # Most GitHub zips have a top-level folder like "repo-main/"
            top_dirs = set()
            for n in names:
                parts = n.split("/")
                if len(parts) > 1:
                    top_dirs.add(parts[0])

            dest = SKILLS_DIR / skill_name
            if dest.exists():
                shutil.rmtree(dest)

            if len(top_dirs) == 1:
                # Single top-level dir — extract its contents directly
                top_dir = list(top_dirs)[0]
                dest.mkdir(parents=True, exist_ok=True)
                for member in zf.namelist():
                    if member.startswith(top_dir + "/") and not member.endswith("/"):
                        rel_path = member[len(top_dir) + 1:]
                        if rel_path:
                            target = dest / rel_path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(member) as src, open(target, "wb") as dst:
                                dst.write(src.read())
            else:
                zf.extractall(dest)

        # Register source in DB
        add_skill_source(skill_name, source_type, source_uri)

        return {
            "success": True,
            "name": skill_name,
            "source": source_uri,
            "path": str(dest),
            "files": len(list(dest.rglob("*"))),
        }

    except zipfile.BadZipFile:
        return {"error": "Downloaded file is not a valid zip archive"}
    except Exception as e:
        return {"error": str(e)}


def uninstall_skill(name: str) -> dict:
    """Remove an installed skill and its source record."""
    import shutil
    from memory import remove_skill_source

    SKILLS_DIR = Path(os.getenv("SYNTHCLAW_BASE_DIR", "/opt/agent")) / ".skills"
    skill_path = SKILLS_DIR / name

    removed_files = False
    if skill_path.exists() and skill_path.is_dir():
        shutil.rmtree(skill_path)
        removed_files = True

    removed_source = remove_skill_source(name)

    if not removed_files and not removed_source:
        return {"error": f"Skill '{name}' not found"}

    return {"success": True, "name": name, "removed_files": removed_files, "removed_source": removed_source}


def list_skills_with_sources() -> dict:
    """List all installed skills with their source information."""
    from memory import list_skill_sources

    SKILLS_DIR = Path(os.getenv("SYNTHCLAW_BASE_DIR", "/opt/agent")) / ".skills"
    sources = list_skill_sources()
    source_map = {s["name"]: s for s in sources}

    skills = []
    if SKILLS_DIR.exists():
        for d in sorted(SKILLS_DIR.iterdir()):
            if d.is_dir():
                source_info = source_map.get(d.name, {})
                skill_md = d / "SKILL.md"
                readme = d / "README.md"
                desc_file = skill_md if skill_md.exists() else (readme if readme.exists() else None)
                desc = ""
                if desc_file:
                    content = desc_file.read_text(encoding="utf-8", errors="ignore")
                    for line in content.split("\n"):
                        line = line.strip().lstrip("#").strip()
                        if line and not line.startswith("---"):
                            desc = line[:100]
                            break
                skills.append({
                    "name": d.name,
                    "description": desc,
                    "source_type": source_info.get("source_type", "unknown"),
                    "source_uri": source_info.get("source_uri", "local"),
                    "auto_update": source_info.get("auto_update", False),
                })

    # Also include sources not yet installed locally
    for name, src in source_map.items():
        if not any(s["name"] == name for s in skills):
            skills.append({
                "name": name,
                "description": "(not installed locally)",
                "source_type": src["source_type"],
                "source_uri": src["source_uri"],
                "auto_update": src.get("auto_update", True),
                "needs_install": True,
            })

    return {"skills": skills, "count": len(skills)}


def reinstall_all_skills() -> dict:
    """Reinstall all skills from their stored sources.
    Called on fresh install when D1 has skill_sources but local .skills is empty.
    """
    from memory import list_skill_sources

    sources = list_skill_sources()
    if not sources:
        return {"message": "No skill sources registered", "installed": 0}

    installed = 0
    errors = []
    for src in sources:
        result = install_skill(src["source_uri"], name=src["name"])
        if result.get("success"):
            installed += 1
        else:
            errors.append(f"{src['name']}: {result.get('error', 'unknown')}")

    return {"installed": installed, "total": len(sources), "errors": errors}


# ── Tool registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "run_command": {
        "fn": run_command,
        "description": "Execute a shell command. Timeout auto-scales: 30s normal, 180s installs, 300s builds. Override with explicit timeout. Returns returncode (0=success, non-zero=FAILED).",
        "params": {"command": "str (required)", "timeout": "int (optional, auto-detected if omitted)"},
    },
    "write_file": {
        "fn": write_file,
        "description": "Write content to a file. Path is relative to /opt/agent/workspace/ unless absolute.",
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
        "description": "Create & start a persistent background systemd service on the droplet",
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
        "description": "Send a file from the server to the user via Telegram (photo, video, audio, document). File is sent after your text reply.",
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
    "web_search": {
        "fn": web_search,
        "description": "Search the web using DuckDuckGo. No API key required. Returns titles, URLs, and snippets.",
        "params": {"query": "str", "max_results": "int (optional, default 8)"},
    },
    "google_search": {
        "fn": google_search,
        "description": "Search Google (high quality results). Falls back to DuckDuckGo if Google API key not configured. Use for factual lookups, docs, current info.",
        "params": {"query": "str", "max_results": "int (optional, default 5)"},
    },
    "scrape_page": {
        "fn": scrape_page,
        "description": "Fetch a web page and return clean readable text (scripts/styles/nav removed).",
        "params": {"url": "str", "max_chars": "int (optional, default 8000)"},
    },
    "scrape_selector": {
        "fn": scrape_selector,
        "description": "Extract elements matching a CSS selector from a web page.",
        "params": {"url": "str", "selector": "str (CSS selector)", "max_results": "int (optional, default 20)"},
    },
    "run_python": {
        "fn": run_python,
        "description": "Execute a Python code snippet in a subprocess. Returns stdout, stderr, returncode.",
        "params": {"code": "str (Python code to run)", "timeout": "int (optional, default 30s)"},
    },
    "exec_code": {
        "fn": exec_code,
        "description": "Run code directly in memory. PREFERRED over write_file+run_command for quick tasks. Use via <tool_call> like any other tool. lang: node (default), python, bash.",
        "params": {"code": "str (code to execute)", "lang": "str (node|python|bash, default: node)", "timeout": "int (optional, default 30s)"},
    },
    "read_csv": {
        "fn": read_csv,
        "description": "Read a CSV file and return its rows as a list of dicts.",
        "params": {"path": "str", "max_rows": "int (optional, default 100)"},
    },
    "query_csv": {
        "fn": query_csv,
        "description": "Filter rows from a CSV where a column equals a given value.",
        "params": {"path": "str", "filter_col": "str (column name)", "filter_val": "str (value to match)", "max_rows": "int (optional, default 100)"},
    },
    "summarize_file": {
        "fn": summarize_file,
        "description": "Read a large file smartly: returns head, middle sample, and tail. Use for files over 5000 chars.",
        "params": {"path": "str", "max_chars": "int (optional, default 4000 — below this, full content returned)"},
    },
    "think": {
        "fn": think,
        "description": "Private reasoning scratchpad. Write out your thinking, plans, or trade-offs. NOT shown to the user. Use before complex multi-step tasks.",
        "params": {"thought": "str (your reasoning, any length)"},
    },
    "ask_user": {
        "fn": ask_user,
        "description": "Pause the task and ask the user a clarifying question. The question is sent to the user; their reply arrives in the next message. Use when you cannot safely proceed without more info.",
        "params": {"question": "str (the question to ask)"},
    },
    "send_email": {
        "fn": send_email,
        "description": "Send an email via SMTP. Requires SMTP credentials stored with store_cred (key='smtp', value='host:port:user:password').",
        "params": {"to": "str", "subject": "str", "body": "str", "from_addr": "str (optional)", "smtp_cred": "str (optional, default 'smtp')"},
    },
    "send_telegram_message": {
        "fn": send_telegram_message,
        "description": "Send a Telegram message to any chat ID (notifications, alerts, DMs to other users). NOT for replying to the current user — use normal replies for that.",
        "params": {"chat_id": "str", "text": "str"},
    },
    "schedule_task": {
        "fn": schedule_task,
        "description": "Create a cron job that runs a command on a schedule. schedule is a cron expression (e.g. '0 * * * *' = hourly, '0 9 * * 1' = Monday 9am).",
        "params": {"name": "str (unique job name)", "command": "str (full shell command)", "schedule": "str (cron expression)"},
    },
    "set_reminder": {
        "fn": set_reminder,
        "description": "Set a timed reminder — sends a Telegram message to the owner at a scheduled time. Use for: 'remind me at 9am', 'daily standup reminder', 'weekly report ping'. Schedule is a cron expression.",
        "params": {"message": "str (reminder text)", "schedule": "str (cron expression, e.g. '0 9 * * *' = daily 9am)", "name": "str (optional, unique identifier)"},
    },
    "list_cron": {
        "fn": list_cron,
        "description": "List all cron jobs, highlighting agent-created ones.",
        "params": {},
    },
    "remove_cron": {
        "fn": remove_cron,
        "description": "Remove a cron job previously created by schedule_task, by its name.",
        "params": {"name": "str (job name used when created)"},
    },
    "tail_logs": {
        "fn": tail_logs,
        "description": "Tail the last N lines from a systemd service log or a log file.",
        "params": {"service": "str (optional, systemd service name)", "path": "str (optional, log file path)", "lines": "int (optional, default 50)"},
    },
    "watch_url": {
        "fn": watch_url,
        "description": "Poll a URL every few seconds until it returns HTTP 200 or timeout is reached. Useful after deploying a service.",
        "params": {"url": "str", "timeout": "int (optional, default 60s)", "interval": "int (optional, default 5s)"},
    },
    "kill_process": {
        "fn": kill_process,
        "description": "Kill a process by name (pkill) or by port (fuser). Use to stop stuck or rogue processes.",
        "params": {"name": "str (optional, process name pattern)", "port": "int (optional, port number)"},
    },
    "sqlite_query": {
        "fn": sqlite_query,
        "description": "Run a SQL query (SELECT or DML) against a SQLite .db file. Returns rows as list of dicts for SELECT.",
        "params": {"db_path": "str", "query": "str (SQL statement)", "params": "list (optional, for parameterized queries)"},
    },
    "redis_get": {
        "fn": redis_get,
        "description": "Get a value from Redis by key.",
        "params": {"key": "str", "host": "str (optional, default 127.0.0.1)", "port": "int (optional, default 6379)"},
    },
    "redis_set": {
        "fn": redis_set,
        "description": "Set a value in Redis. Optionally set TTL in seconds (0 = no expiry).",
        "params": {"key": "str", "value": "str", "ttl": "int (optional, default 0 = no expiry)", "host": "str (optional)", "port": "int (optional)"},
    },
    # ── Dynamic API tools ─────────────────────────────────────────────────
    "register_api": {
        "fn": register_api,
        "description": "Register an external API for future calling. Provide name + base_url + auth_cred (credential name). For known APIs (stripe, vercel, cloudflare, github_api, notion, etc.), just provide name and credential — endpoints are auto-detected.",
        "params": {
            "name": "str (API name, e.g. 'stripe', 'vercel', 'my_custom_api')",
            "base_url": "str (API base URL, or 'auto' for known APIs)",
            "auth_cred": "str (credential name stored via store_cred)",
            "auth_type": "str (optional: bearer, header, query, basic, body)",
            "auth_header": "str (optional, default 'Authorization')",
            "auth_prefix": "str (optional, default 'Bearer ')",
            "endpoints": "str (optional, JSON array of {method, path, desc})",
            "description": "str (optional)",
            "docs_url": "str (optional)",
        },
    },
    "api_call": {
        "fn": api_call,
        "description": "Call a registered external API. Use after register_api. Handles auth automatically.",
        "params": {
            "api": "str (registered API name, e.g. 'stripe')",
            "method": "str (GET, POST, PUT, DELETE, PATCH)",
            "path": "str (endpoint path, e.g. '/charges', '/v9/projects')",
            "body": "str (optional, JSON body for POST/PUT)",
            "query_params": "str (optional, e.g. 'limit=10&status=active')",
            "headers": "str (optional, extra headers as JSON object)",
        },
    },
    "list_apis": {
        "fn": list_apis,
        "description": "List all registered dynamic APIs and their endpoints. Shows what external services the agent can call.",
        "params": {},
    },
    "composio_execute": {
        "fn": composio_execute,
        "description": "Execute any Composio tool (1000+ apps: Gmail, Slack, GitHub, Notion, etc). Use /connectors to connect apps first.",
        "params": {
            "tool_slug": "str (e.g. GMAIL_SEND_EMAIL, GITHUB_CREATE_ISSUE, SLACK_SEND_MESSAGE)",
            "args": "str (JSON arguments for the tool)",
            "connected_account_id": "str (optional)",
        },
    },
    # ── Skill management tools ─────────────────────────────────────────
    "install_skill": {
        "fn": install_skill,
        "description": "Install a skill from a source (GitHub repo, URL to zip, or search term). Registered in D1 for auto-reinstall on fresh installs.",
        "params": {
            "source": "str (github:owner/repo, clawhub:@owner/name, https://url.zip, or search term)",
            "name": "str (optional — override skill name)",
        },
    },
    "uninstall_skill": {
        "fn": uninstall_skill,
        "description": "Remove an installed skill and its source record.",
        "params": {"name": "str (skill name to remove)"},
    },
    "list_skills_with_sources": {
        "fn": list_skills_with_sources,
        "description": "List all installed skills with their source information (for reinstall). Shows which are from D1 vs local-only.",
        "params": {},
    },
    "reinstall_all_skills": {
        "fn": reinstall_all_skills,
        "description": "Reinstall all skills from stored sources (D1). Use on fresh install to restore skill set.",
        "params": {},
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
    """Generate compact tool listing for system prompt. Optimized to minimize tokens."""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        # Compact format: name(param1, param2) — one-line description
        params = list(info["params"].keys())
        params_str = ", ".join(params) if params else ""
        # Truncate description to first sentence
        desc = info["description"].split(".")[0].strip()
        lines.append(f"{name}({params_str}) — {desc}")
    return "\n".join(lines)
