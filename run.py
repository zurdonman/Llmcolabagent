#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
APP_LOG = LOG_DIR / "app.log"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        return any(getattr(s, 'isatty', lambda: False)() for s in self.streams)


def _setup_logging():
    try:
        f = open(APP_LOG, "a", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, f)
        sys.stderr = _Tee(sys.__stderr__, f)
        print(f"[setup] Log de app: {APP_LOG}")
    except Exception as e:
        print(f"[setup] No se pudo inicializar log de app: {e}")


def ensure_deps():
    required = ["fastapi", "uvicorn", "httpx"]
    missing = []
    for mod in required:
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if not missing:
        return
    req = ROOT / "requirements.txt"
    print(f"[setup] Instalando dependencias: {', '.join(missing)} ...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        # retry with --break-system-packages for managed envs like Colab containers
        cmd2 = cmd + ["--break-system-packages"]
        print("[setup] Reintentando instalación...")
        r2 = subprocess.run(cmd2)
        if r2.returncode != 0:
            print("[setup] No se pudieron instalar dependencias automáticamente.")
            print(f"[setup] Ejecuta manualmente: {sys.executable} -m pip install -r requirements.txt")
            sys.exit(1)


if __name__ == "__main__":
    _setup_logging()
    ensure_deps()
    from llm_remote.launcher import main
    main()
