import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path


def _is_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def _bin_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tunnel] Descargando {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "llm-remote/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def ensure_cloudflared() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found

    system = platform.system().lower()
    machine = platform.machine().lower()
    bdir = _bin_dir()

    if system == "linux":
        arch = "amd64" if machine in ("x86_64", "amd64") else "arm64" if machine in ("aarch64", "arm64") else None
        if not arch:
            raise RuntimeError(f"Arquitectura no soportada para cloudflared: {machine}")
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
        dest = bdir / "cloudflared"
        if not dest.exists():
            _download(url, dest)
            dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
        return str(dest)

    if system == "darwin":
        arch = "amd64.tgz" if machine in ("x86_64", "amd64") else "arm64.tgz"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-{arch}"
        dest = bdir / "cloudflared"
        if not dest.exists():
            with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _download(url, tmp_path)
                with tarfile.open(tmp_path, "r:gz") as tar:
                    member = tar.getmember("cloudflared")
                    tar.extract(member, path=bdir)
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
            finally:
                tmp_path.unlink(missing_ok=True)
        return str(dest)

    if system == "windows":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        dest = bdir / "cloudflared.exe"
        if not dest.exists():
            _download(url, dest)
        return str(dest)

    raise RuntimeError(f"Sistema no soportado para cloudflared auto-install: {system}")


def start_cloudflared(local_port: int) -> tuple[subprocess.Popen, str]:
    """Start a cloudflared quick tunnel and return (process, public_url)."""
    bin_path = ensure_cloudflared()
    cmd = [bin_path, "tunnel", "--url", f"http://127.0.0.1:{local_port}", "--no-autoupdate"]
    print(f"[tunnel] Iniciando cloudflared: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url = None
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 60
    lines = []
    while time.time() < deadline:
        if proc.poll() is not None:
            rest = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"cloudflared salió prematuramente.\n{''.join(lines)}\n{rest}")
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            lines.append(line)
            m = pattern.search(line)
            if m:
                url = m.group(0)
                break
        else:
            time.sleep(0.2)

    if not url:
        proc.terminate()
        raise RuntimeError(f"No se pudo obtener URL de cloudflared.\n{''.join(lines)}")

    # Drain output in background to prevent blocking
    def _drain():
        try:
            for l in proc.stdout:
                if "trycloudflare.com" in l or "error" in l.lower() or "ERR" in l:
                    print(f"[cloudflared] {l.rstrip()}")
        except Exception:
            pass

    import threading
    threading.Thread(target=_drain, daemon=True).start()
    return proc, url


def start_ngrok(local_port: int, token: str = "") -> tuple[object, str]:
    """Start ngrok tunnel. Requires pyngrok or ngrok installed."""
    try:
        from pyngrok import ngrok, conf
    except Exception as e:
        raise RuntimeError(
            "ngrok seleccionado pero pyngrok no está instalado. Instala con: pip install pyngrok"
        ) from e

    if token:
        ngrok.set_auth_token(token)
    # Close stale tunnels
    try:
        ngrok.kill()
    except Exception:
        pass
    tunnel = ngrok.connect(local_port, "http")
    url = tunnel.public_url
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return tunnel, url


def start_tunnel(kind: str, local_port: int, ngrok_token: str = ""):
    if kind == "none":
        return None, None
    if kind == "cloudflared":
        return start_cloudflared(local_port)
    if kind == "ngrok":
        return start_ngrok(local_port, ngrok_token)
    raise ValueError(f"Túnel desconocido: {kind}")
