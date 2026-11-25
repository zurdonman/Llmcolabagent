import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path


def _bin_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _models_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path, desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[backend] Ya existe: {dest}")
        return
    label = desc or url
    print(f"[backend] Descargando {label} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "llm-remote/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            total = r.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            chunk = 1024 * 1024
            last_report = 0.0
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                now = time.time()
                if now - last_report >= 5:
                    if total:
                        pct = downloaded * 100 / total
                        print(f"[backend]   {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)")
                    else:
                        print(f"[backend]   {downloaded/1e6:.1f} MB")
                    last_report = now
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    print(f"[backend] Descarga completada: {dest}")


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "llm-remote/1.0", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def ensure_llama_server(binary_hint: str = "") -> str:
    """Return path to llama-server binary, downloading if needed."""
    if binary_hint:
        p = Path(binary_hint)
        if p.exists():
            return str(p)
        found = shutil.which(binary_hint)
        if found:
            return found
        raise FileNotFoundError(f"llama-server no encontrado: {binary_hint}")

    for name in ("llama-server", "llama-server.exe"):
        found = shutil.which(name)
        if found:
            return found

    bdir = _bin_dir()
    system = platform.system().lower()
    machine = platform.machine().lower()
    exe = "llama-server.exe" if system == "windows" else "llama-server"
    local = bdir / exe
    if local.exists():
        return str(local)

    # Try to fetch latest release from GitHub API
    print("[backend] Buscando release más reciente de llama.cpp...")
    try:
        rel = _http_json("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
    except Exception as e:
        raise RuntimeError(
            "No se pudo consultar la release de llama.cpp. "
            "Especifica --llama-binary o instala llama-server manualmente."
        ) from e

    tag = rel.get("tag_name", "")
    assets = rel.get("assets", [])
    names = [a["name"] for a in assets]

    def pick(candidates):
        for c in candidates:
            for a in assets:
                if a["name"] == c:
                    return a
        for c in candidates:
            for a in assets:
                if c in a["name"]:
                    return a
        return None

    asset = None
    if system == "linux" and machine in ("x86_64", "amd64"):
        # Prefer CUDA build if nvidia-smi present, else ubuntu/cpu
        if _has_nvidia_gpu():
            asset = pick([n for n in names if "ubuntu" in n and "x64" in n and "cu" in n and n.endswith(".tar.gz")])
        if not asset:
            asset = pick([n for n in names if "ubuntu" in n and "x64" in n and n.endswith(".tar.gz") and "cu" not in n])
        if not asset:
            asset = pick([n for n in names if "linux" in n and "x64" in n and n.endswith(".tar.gz")])
    elif system == "darwin":
        if machine in ("arm64", "aarch64"):
            asset = pick([n for n in names if "macos" in n and "arm64" in n and n.endswith(".tar.gz")])
        else:
            asset = pick([n for n in names if "macos" in n and "x64" in n and n.endswith(".tar.gz")])
    elif system == "windows":
        if _has_nvidia_gpu():
            asset = pick([n for n in names if "win" in n and "cu" in n and "x64" in n and n.endswith(".zip")])
        if not asset:
            asset = pick([n for n in names if "win" in n and "cpu" in n and "x64" in n and n.endswith(".zip")])

    if not asset:
        raise RuntimeError(
            f"No se encontró un asset de llama-server para {system}/{machine} en release {tag}. "
            f"Assets disponibles: {names[:20]}"
        )

    url = asset["browser_download_url"]
    archive_path = bdir / asset["name"]
    _download(url, archive_path, desc=f"llama.cpp {tag} ({asset['name']})")

    print(f"[backend] Extrayendo {archive_path.name} ...")
    extract_dir = bdir / f"llama-{tag}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_dir)
    else:
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(extract_dir)

    # find llama-server inside
    candidates = list(extract_dir.rglob(exe))
    if not candidates:
        # sometimes named 'server' in old builds
        candidates = list(extract_dir.rglob("server")) + list(extract_dir.rglob("server.exe"))
    if not candidates:
        raise RuntimeError(f"No se encontró {exe} dentro de {archive_path}")

    src = candidates[0]
    # copy binary and sibling shared libs
    target_dir = bdir / "llama-bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    # copy all files from same directory (shared libs) 
    for f in src.parent.iterdir():
        if f.is_file():
            dst = target_dir / f.name
            shutil.copy2(f, dst)
            if os.name != "nt":
                dst.chmod(dst.stat().st_mode | stat.S_IEXEC)
    final = target_dir / src.name
    # also create/link standard name
    std = bdir / exe
    if std != final:
        try:
            if std.exists() or std.is_symlink():
                std.unlink()
            std.symlink_to(final)
        except Exception:
            shutil.copy2(final, std)
            if os.name != "nt":
                std.chmod(std.stat().st_mode | stat.S_IEXEC)
    if os.name != "nt":
        final.chmod(final.stat().st_mode | stat.S_IEXEC)
    print(f"[backend] llama-server listo: {final}")
    return str(final)


def resolve_model_path(model: str) -> tuple[str, bool]:
    """Return (model_arg, is_hf_repo). Downloads URLs to models/."""
    if not model:
        raise ValueError("Debes especificar --model (ruta GGUF, URL, o repo HF como org/repo:quant)")
    # URL
    if model.startswith("http://") or model.startswith("https://"):
        name = model.rstrip("/").split("/")[-1].split("?")[0] or "model.gguf"
        if not name.endswith(".gguf"):
            name += ".gguf"
        dest = _models_dir() / name
        _download(model, dest, desc=name)
        return str(dest), False
    # Local path
    p = Path(model).expanduser()
    if p.exists():
        return str(p.resolve()), False
    # HF repo style: owner/repo or owner/repo:file.gguf or owner/repo:quant
    # llama-server supports -hf <repo> [-hff <file>]
    if "/" in model and not model.startswith(".") and " " not in model:
        return model, True
    raise FileNotFoundError(
        f"Modelo no encontrado: {model}. "
        "Usa una ruta local .gguf, una URL directa, o un repo de HuggingFace (ej: bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M)."
    )


def start_llama_server(cfg, internal_port: int, log_file) -> subprocess.Popen:
    binary = ensure_llama_server(cfg.llama_binary)
    model_arg, is_hf = resolve_model_path(cfg.model)

    cmd = [binary, "--host", "127.0.0.1", "--port", str(internal_port), "-c", str(cfg.n_ctx)]

    if is_hf:
        # support owner/repo:file_or_quant
        if ":" in model_arg:
            repo, selector = model_arg.rsplit(":", 1)
            cmd += ["-hf", repo]
            if selector.endswith(".gguf"):
                cmd += ["-hff", selector]
            else:
                cmd += ["-hff", selector]
        else:
            cmd += ["-hf", model_arg]
    else:
        cmd += ["-m", model_arg]

    if cfg.n_gpu_layers != 0:
        cmd += ["-ngl", str(cfg.n_gpu_layers)]
    if cfg.cache_type_k:
        cmd += ["--cache-type-k", cfg.cache_type_k]
    if cfg.cache_type_v:
        cmd += ["--cache-type-v", cfg.cache_type_v]
    if cfg.flash_attn:
        cmd += ["-fa"]
    if cfg.n_batch and cfg.n_batch > 0:
        cmd += ["-b", str(cfg.n_batch)]
    if cfg.n_threads and cfg.n_threads > 0:
        cmd += ["-t", str(cfg.n_threads)]
    if not cfg.ctx_shift:
        cmd += ["--no-context-shift"]
    if cfg.defrag_thold >= 0:
        cmd += ["--defrag-thold", str(cfg.defrag_thold)]
    if cfg.keep and cfg.keep > 0:
        cmd += ["--keep", str(cfg.keep)]

    cmd += cfg.extra_args

    print(f"[backend] Iniciando llama-server: {' '.join(cmd)}")
    env = os.environ.copy()
    # Make sure sibling libs are found
    bin_dir = str(Path(binary).parent)
    if platform.system().lower() == "linux":
        env["LD_LIBRARY_PATH"] = bin_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    elif platform.system().lower() == "darwin":
        env["DYLD_LIBRARY_PATH"] = bin_dir + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc


# ---------------- Ollama ----------------

def ensure_ollama_installed() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    system = platform.system().lower()
    if system == "linux":
        print("[backend] Ollama no encontrado. Instalando...")
        # Official install script
        r = subprocess.run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"], check=False)
        if r.returncode != 0:
            raise RuntimeError("Falló la instalación de Ollama. Instálalo manualmente: https://ollama.com")
        found = shutil.which("ollama")
        if found:
            return found
        # common location
        for p in ["/usr/local/bin/ollama", "/usr/bin/ollama"]:
            if Path(p).exists():
                return p
        raise RuntimeError("Ollama instalado pero no encontrado en PATH.")
    raise RuntimeError(
        "Ollama no está instalado. Instálalo desde https://ollama.com y vuelve a ejecutar."
    )


def start_ollama_serve(log_file, ollama_host: str = "127.0.0.1:11434") -> subprocess.Popen | None:
    """Start `ollama serve` if not already responding. Returns process or None if already running."""
    import urllib.error
    base = f"http://{ollama_host}"
    try:
        req = urllib.request.Request(f"{base}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.status == 200:
                print(f"[backend] Ollama ya está corriendo en {ollama_host}")
                return None
    except Exception:
        pass

    ollama_bin = ensure_ollama_installed()
    print("[backend] Iniciando ollama serve ...")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = ollama_host
    proc = subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # wait for readiness
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("ollama serve salió prematuramente. Revisa el log.")
        try:
            req = urllib.request.Request(f"{base}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as r:
                if r.status == 200:
                    print("[backend] Ollama listo.")
                    return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("Ollama no respondió a tiempo.")


def ollama_pull(model: str) -> None:
    ollama_bin = ensure_ollama_installed()
    print(f"[backend] Descargando modelo Ollama: {model} (ollama pull) ...")
    r = subprocess.run([ollama_bin, "pull", model], check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Falló `ollama pull {model}`")


def start_backend(cfg, internal_port: int, log_file):
    """Start the selected backend. Returns (process_or_none, backend_type, backend_base_url)."""
    if cfg.backend == "llama.cpp":
        proc = start_llama_server(cfg, internal_port, log_file)
        return proc, "llama.cpp", f"http://127.0.0.1:{internal_port}"
    if cfg.backend == "ollama":
        proc = start_ollama_serve(log_file, ollama_host=cfg.ollama_host)
        if not cfg.model:
            raise ValueError("Debes especificar --model para ollama (ej: qwen2.5:1.5b)")
        ollama_pull(cfg.model)
        # Ollama serves OpenAI-compatible at /v1
        return proc, "ollama", f"http://{cfg.ollama_host}"
    raise ValueError(f"Backend desconocido: {cfg.backend}")


def list_ollama_models(ollama_host: str = "127.0.0.1:11434") -> list[dict]:
    """Return local Ollama models from /api/tags."""
    url = f"http://{ollama_host}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-remote/1.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return []
    models = data.get("models", []) if isinstance(data, dict) else []
    out = []
    for m in models:
        out.append({
            "name": m.get("name") or m.get("model"),
            "size": m.get("size"),
            "size_gb": round(m.get("size", 0) / (1024**3), 2) if m.get("size") else None,
            "modified_at": m.get("modified_at"),
            "details": m.get("details", {}),
        })
    return out


def ollama_running(ollama_host: str = "127.0.0.1:11434") -> bool:
    try:
        req = urllib.request.Request(f"http://{ollama_host}/api/tags", headers={"User-Agent": "llm-remote/1.0"})
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def stop_process(proc: subprocess.Popen | None):
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _ollama_models_dir() -> Path | None:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    candidates = [
        Path.home() / ".ollama" / "models",
        Path("/usr/share/ollama/.ollama/models"),
        Path("/var/lib/ollama/.ollama/models"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def list_ollama_gguf_models() -> list[dict]:
    """Scan Ollama manifests and return models whose model layer is a local GGUF blob."""
    models_dir = _ollama_models_dir()
    if not models_dir:
        return []
    manifests_dir = models_dir / "manifests"
    blobs_dir = models_dir / "blobs"
    if not manifests_dir.exists() or not blobs_dir.exists():
        return []
    results = []
    for manifest_path in manifests_dir.rglob("*"):
        if not manifest_path.is_file():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        layers = data.get("layers", []) if isinstance(data, dict) else []
        model_digest = None
        for layer in layers:
            mt = (layer.get("mediaType") or "").lower()
            if "model" in mt:
                model_digest = layer.get("digest")
                break
        if not model_digest:
            continue
        # digest like sha256:abcdef
        digest = model_digest.replace(":", "-")
        blob_path = blobs_dir / digest
        if not blob_path.exists():
            # sometimes sha256:...
            blob_path = blobs_dir / model_digest.replace(":", "-")
        if not blob_path.exists():
            continue
        size = blob_path.stat().st_size
        # Build model name from manifest path relative to manifests dir
        try:
            rel = manifest_path.relative_to(manifests_dir)
            parts = rel.parts
            # typical: registry.ollama.ai/library/<name>/<tag> or registry.ollama.ai/<user>/<name>/<tag>
            if len(parts) >= 4 and parts[0] == "registry.ollama.ai":
                if parts[1] == "library":
                    name = f"{parts[2]}:{parts[3]}" if len(parts) > 3 else parts[2]
                else:
                    name = f"{parts[1]}/{parts[2]}:{parts[3]}" if len(parts) > 3 else "/".join(parts[1:])
            else:
                name = ":".join(parts[-2:]) if len(parts) >= 2 else str(rel)
        except Exception:
            name = manifest_path.name
        results.append({
            "name": name,
            "path": str(blob_path),
            "size": size,
            "size_gb": round(size / (1024**3), 2),
            "source": "ollama",
        })
    # dedupe by path
    seen = set()
    uniq = []
    for r in results:
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        uniq.append(r)
    uniq.sort(key=lambda x: x["name"])
    return uniq
