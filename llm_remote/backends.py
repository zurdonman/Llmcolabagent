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
from typing import Callable, Optional

_progress_cb: Optional[Callable[[dict], None]] = None


def set_progress_callback(cb: Optional[Callable[[dict], None]]):
    global _progress_cb
    _progress_cb = cb


def _report_progress(**kwargs):
    if _progress_cb:
        try:
            _progress_cb(kwargs)
        except Exception:
            pass


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
        _report_progress(active=False, label=desc or dest.name, percent=100, downloaded=dest.stat().st_size, total=dest.stat().st_size)
        return
    label = desc or url
    print(f"[backend] Descargando {label} ...")
    _report_progress(active=True, label=label, percent=0, downloaded=0, total=None)
    req = urllib.request.Request(url, headers={"User-Agent": "llm-remote/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
            total = r.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            chunk = 1024 * 1024
            last_report = 0.0
            t0 = time.time()
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                now = time.time()
                if now - last_report >= 1:
                    pct = (downloaded * 100 / total) if total else 0
                    speed = downloaded / max(now - t0, 0.001)
                    msg = f"{downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)" if total else f"{downloaded/1e6:.1f} MB"
                    print(f"[backend]   {msg}")
                    _report_progress(active=True, label=label, percent=round(pct,1), downloaded=downloaded, total=total, speed=speed)
                    last_report = now
        tmp.replace(dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
    size = dest.stat().st_size if dest.exists() else downloaded
    _report_progress(active=False, label=label, percent=100, downloaded=size, total=size)
    print(f"[backend] Descarga completada: {dest}")


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "llm-remote/1.0", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def log_gpu_diagnostics():
    """Log GPU/CUDA/Vulkan diagnostics to help debug offload issues."""
    import subprocess as _sp
    lines = []
    cmds = [
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        ["nvcc", "--version"],
        ["vulkaninfo", "--summary"],
        ["ldconfig", "-p"],
    ]
    for cmd in cmds:
        try:
            r = _sp.run(cmd, capture_output=True, text=True, timeout=15)
            out = (r.stdout or "") + (r.stderr or "")
            out = out.strip()
            if out:
                lines.append(f"$ {' '.join(cmd)}\n{out[:2000]}")
        except FileNotFoundError:
            lines.append(f"$ {' '.join(cmd)}\n(no encontrado)")
        except Exception as e:
            lines.append(f"$ {' '.join(cmd)}\n(error: {e})")
    msg = "\n".join(lines)
    print("[diag] Diagnóstico GPU:\n" + msg)
    return msg


def ensure_vulkan_runtime():
    """Try to install Vulkan runtime on Linux if missing, and ensure NVIDIA ICD exists."""
    if platform.system().lower() != "linux":
        return
    import subprocess as _sp, shutil as _sh, json as _json, glob as _glob

    def _has_nvidia_vulkan_icd():
        for d in ["/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/local/share/vulkan/icd.d"]:
            p = Path(d)
            if p.exists() and any("nvidia" in f.name.lower() for f in p.glob("*.json")):
                return True
        # also check ICD env
        for k in ["VK_ICD_FILENAMES", "VK_DRIVER_FILES"]:
            if os.environ.get(k):
                return True
        return False

    has_libvulkan = Path("/usr/lib/x86_64-linux-gnu/libvulkan.so.1").exists() or Path("/usr/lib64/libvulkan.so.1").exists()
    has_vulkaninfo = bool(_sh.which("vulkaninfo"))
    has_icd = _has_nvidia_vulkan_icd()

    if not has_libvulkan or not has_vulkaninfo:
        if _sh.which("apt-get"):
            print("[backend] Instalando Vulkan runtime (libvulkan1, vulkan-tools)...")
            try:
                _sp.run(["apt-get", "update", "-qq"], check=False, capture_output=True, text=True, timeout=120)
                _sp.run(["apt-get", "install", "-y", "-qq", "libvulkan1", "vulkan-tools"],
                        check=False, capture_output=True, text=True, timeout=300)
            except Exception as e:
                print(f"[backend] No se pudo instalar Vulkan runtime: {e}")
        else:
            print("[backend] No se encontró apt-get para instalar Vulkan runtime.")

    # Ensure NVIDIA Vulkan ICD exists (Colab sometimes missing it)
    if not _has_nvidia_vulkan_icd() and _has_nvidia_gpu():
        print("[backend] NVIDIA Vulkan ICD no encontrado; intentando crear...")
        icd_dir = Path("/usr/share/vulkan/icd.d")
        try:
            icd_dir.mkdir(parents=True, exist_ok=True)
            # Find nvidia lib
            lib_candidates = _glob.glob("/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*") + \
                             _glob.glob("/usr/lib64/libGLX_nvidia.so*") + \
                             _glob.glob("/usr/lib/x86_64-linux-gnu/nvidia*/libGLX_nvidia.so*")
            lib_path = "libGLX_nvidia.so.0"
            for c in lib_candidates:
                lib_path = c
                break
            icd = {"file_format_version": "1.0.0", "ICD": {"library_path": lib_path, "api_version": "1.3.0"}}
            icd_file = icd_dir / "nvidia_icd.json"
            icd_file.write_text(_json.dumps(icd), encoding="utf-8")
            print(f"[backend] NVIDIA Vulkan ICD creado: {icd_file} -> {lib_path}")
        except Exception as e:
            print(f"[backend] No se pudo crear NVIDIA Vulkan ICD: {e}")

    if _sh.which("vulkaninfo"):
        print("[backend] Vulkan runtime listo.")
    else:
        print("[backend] Vulkan runtime instalado parcialmente (vulkaninfo no disponible).")


def _detect_cuda_version() -> tuple[int, int] | None:
    """Return (major, minor) CUDA version if detectable, else None."""
    import re, subprocess as _sp
    # Try nvidia-smi
    try:
        r = _sp.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", r.stdout + r.stderr)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    # Try nvcc
    try:
        r = _sp.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        m = re.search(r"release\s*(\d+)\.(\d+)", r.stdout + r.stderr)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass
    # Try libcudart
    for p in ["/usr/local/cuda/lib64", "/usr/local/cuda/lib", "/usr/lib/x86_64-linux-gnu"]:
        try:
            for f in Path(p).glob("libcudart.so.*"):
                m = re.search(r"libcudart\.so\.(\d+)\.(\d+)", f.name)
                if m:
                    return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
    return None


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

    # Find a release with a matching binary asset (skip tag-only releases like v1.0.0)
    print("[backend] Buscando release de llama.cpp con binarios...")
    _report_progress(active=True, label="Consultando releases de llama.cpp", percent=0)
    try:
        releases = _http_json("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10")
        if isinstance(releases, dict):
            releases = [releases]
    except Exception as e:
        _report_progress(active=False, label="Error consultando llama.cpp", percent=0)
        raise RuntimeError(
            "No se pudo consultar releases de llama.cpp. "
            "Especifica --llama-binary o instala llama-server manualmente."
        ) from e

    def pick_asset(assets, system, machine):
        names = [a["name"] for a in assets]
        def by_name(preds):
            for pred in preds:
                for a in assets:
                    n = a["name"].lower()
                    if pred(n):
                        return a
            return None
        if system == "linux" and machine in ("x86_64", "amd64"):
            if _has_nvidia_gpu():
                cuda_ver = _detect_cuda_version()
                print(f"[backend] CUDA detectada: {cuda_ver[0]}.{cuda_ver[1]}" if cuda_ver else "[backend] GPU NVIDIA detectada (versión CUDA desconocida)")
                all_cuda = [a for a in assets if a["name"].lower().endswith(".tar.gz") and "ubuntu" in a["name"].lower() and "x64" in a["name"].lower() and "cuda" in a["name"].lower()]
                def cuda_build_ver(name):
                    import re
                    m = re.search(r"cuda-(\d+)\.(\d+)", name.lower())
                    return (int(m.group(1)), int(m.group(2))) if m else (0,0)
                def is_cudart(name):
                    return "cudart" in name.lower()

                # Prefer build <= driver CUDA. Prefer non-cudart (contains server binary); cudart is companion libs.
                candidates = []
                if cuda_ver:
                    leq = sorted([a for a in all_cuda if cuda_build_ver(a["name"]) <= cuda_ver], key=lambda a: (cuda_build_ver(a["name"]), 0 if not is_cudart(a["name"]) else 1), reverse=True)
                    seen = set()
                    for a in leq:
                        if a["name"] not in seen:
                            seen.add(a["name"])
                            candidates.append(a)
                    gt = sorted([a for a in all_cuda if cuda_build_ver(a["name"]) > cuda_ver], key=lambda a: (cuda_build_ver(a["name"]), 0 if not is_cudart(a["name"]) else 1))
                    for a in gt:
                        if a["name"] not in seen:
                            seen.add(a["name"])
                            candidates.append(a)
                else:
                    candidates = sorted(all_cuda, key=lambda a: (0 if not is_cudart(a["name"]) else 1))

                for a in candidates:
                    bv = cuda_build_ver(a["name"])
                    kind = "cudart" if is_cudart(a["name"]) else "cuda"
                    print(f"[backend] Candidato {kind}: {a['name']} (build cuda {bv[0]}.{bv[1]})")
                    return a

                # Vulkan fallback only if no CUDA builds
                vulkan = by_name([lambda n: n.endswith(".tar.gz") and "ubuntu" in n and "x64" in n and "vulkan" in n])
                if vulkan:
                    print(f"[backend] Candidato Vulkan (sin builds CUDA): {vulkan['name']}")
                    return vulkan

            a = by_name([
                lambda n: n.endswith(".tar.gz") and "ubuntu" in n and "x64" in n and all(k not in n for k in ["cuda","vulkan","rocm","sycl","openvino","opencl","cann"]),
            ])
            if a: return a
            a = by_name([lambda n: n.endswith(".tar.gz") and "linux" in n and "x64" in n and "cuda" not in n])
            return a
        if system == "darwin":
            arch = "arm64" if machine in ("arm64","aarch64") else "x64"
            return by_name([lambda n, arch=arch: n.endswith(".tar.gz") and "macos" in n and arch in n])
        if system == "windows":
            if _has_nvidia_gpu():
                a = by_name([lambda n: n.endswith(".zip") and "win" in n and "x64" in n and "cuda" in n and "cudart" not in n])
                if a: return a
            return by_name([
                lambda n: n.endswith(".zip") and "win" in n and "x64" in n and "cpu" in n,
                lambda n: n.endswith(".zip") and "win" in n and "x64" in n and all(k not in n for k in ["cuda","vulkan","rocm","sycl","openvino","opencl","cann"]),
            ])
        return None

    asset = None
    tag = ""
    for rel in releases:
        tag = rel.get("tag_name", "")
        assets = rel.get("assets", []) or []
        print(f"[backend] Revisando release {tag} ({len(assets)} assets)...")
        asset = pick_asset(assets, system, machine)
        if asset:
            print(f"[backend] Release seleccionada: {tag} -> {asset['name']}")
            break
    if not asset:
        raise RuntimeError(
            f"No se encontró un asset de llama-server para {system}/{machine} en releases recientes. "
            "Especifica --llama-binary o instala llama-server manualmente."
        )

    # Find companion cudart asset if we picked a cuda build (for runtime libs)
    cudart_asset = None
    if "cuda" in asset["name"].lower() and "cudart" not in asset["name"].lower():
        # look for matching cudart in same release assets
        for a in assets:
            n = a["name"].lower()
            if "cudart" in n and "cuda" in n and n.endswith(".tar.gz"):
                # match same cuda version and arch
                import re
                v1 = re.search(r"cuda-(\d+\.\d+)", asset["name"].lower())
                v2 = re.search(r"cuda-(\d+\.\d+)", n)
                if v1 and v2 and v1.group(1) == v2.group(1) and ("x64" in n) == ("x64" in asset["name"].lower()):
                    cudart_asset = a
                    break
        if cudart_asset:
            print(f"[backend] También se usará cudart: {cudart_asset['name']}")

    # If Vulkan build selected, ensure Vulkan runtime is present
    if "vulkan" in asset["name"].lower():
        ensure_vulkan_runtime()

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

    # Also extract cudart companion if present
    if cudart_asset:
        ca_path = bdir / cudart_asset["name"]
        _download(cudart_asset["browser_download_url"], ca_path, desc=f"cudart {tag} ({cudart_asset['name']})")
        print(f"[backend] Extrayendo {ca_path.name} ...")
        ca_extract = bdir / f"llama-{tag}-cudart"
        ca_extract.mkdir(parents=True, exist_ok=True)
        if ca_path.name.endswith(".zip"):
            with zipfile.ZipFile(ca_path, "r") as z:
                z.extractall(ca_extract)
        else:
            with tarfile.open(ca_path, "r:*") as t:
                t.extractall(ca_extract)
        # copy cudart libs into main extract dir (so they end up next to llama-server)
        for f in ca_extract.rglob("*"):
            if f.is_file() and (".so" in f.name or f.suffix in (".dll", ".dylib")):
                dest_f = extract_dir / f.name
                if not dest_f.exists():
                    shutil.copy2(f, dest_f)

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
    # also copy any cudart/shared libs extracted at extract_dir root
    for f in extract_dir.iterdir():
        if f.is_file() and (".so" in f.name or f.suffix in (".dll",".dylib")):
            dst = target_dir / f.name
            if not dst.exists():
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
    # Diagnostic: list devices
    try:
        import subprocess as _sp
        env = os.environ.copy()
        bdir_lib = str(final.parent)
        if platform.system().lower() == "linux":
            env["LD_LIBRARY_PATH"] = bdir_lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        for args in [[str(final), "--list-devices"], [str(final), "--version"]]:
            try:
                r = _sp.run(args, capture_output=True, text=True, timeout=30, env=env)
                out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
                if out:
                    print(f"[diag] {' '.join(args[-1:])}:\n{out[:3000]}")
            except Exception as e:
                print(f"[diag] {' '.join(args)} falló: {e}")
    except Exception as e:
        print(f"[diag] No se pudo ejecutar diagnóstico de llama-server: {e}")
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
        # Download HF GGUF ourselves for real-time progress, then load locally
        if ":" in model_arg:
            repo, selector = model_arg.rsplit(":", 1)
            filename = selector.strip()
        else:
            repo, filename = model_arg, ""
        if filename and filename.lower().endswith(".gguf"):
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            dest = _models_dir() / filename.replace("/", "_")
            print(f"[backend] Descargando modelo desde HuggingFace: {repo}/{filename}")
            _download(url, dest, desc=filename)
            cmd += ["-m", str(dest)]
        else:
            # Fallback: let llama-server handle it
            if ":" in model_arg:
                repo, selector = model_arg.rsplit(":", 1)
                cmd += ["-hf", repo, "-hff", selector]
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
        cmd += ["--flash-attn", "on"]
    if cfg.n_batch and cfg.n_batch > 0:
        cmd += ["-b", str(cfg.n_batch)]
    if cfg.n_threads and cfg.n_threads > 0:
        cmd += ["-t", str(cfg.n_threads)]
    if cfg.n_parallel and cfg.n_parallel > 0:
        cmd += ["--parallel", str(cfg.n_parallel)]
    if cfg.jinja:
        cmd += ["--jinja"]
    if not cfg.ctx_shift:
        cmd += ["--no-context-shift"]
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
