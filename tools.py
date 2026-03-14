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
            "stdout": _smart_truncate(r.stdout),
            "stderr": _smart_truncate(r.stderr, 1500),
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


def scrape_page(url: str, max_chars: int = 8000) -> dict:
    """Fetch a web page and return clean readable text (scripts/styles/nav removed)."""
    try:
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgentBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
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
