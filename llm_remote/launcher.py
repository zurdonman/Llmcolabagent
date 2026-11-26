import os
import signal
import sys
import time
from pathlib import Path

import uvicorn

from .config import Config, build_parser, config_from_args
from .tunnel import start_tunnel
from .webapp import create_app, get_state


def _print_banner(cfg: Config, public_url: str | None):
    local = f"http://127.0.0.1:{cfg.port}"
    ui = public_url or local
    print("\n" + "=" * 70)
    print("  LLM REMOTE - PANEL WEB LISTO")
    print("=" * 70)
    print(f"  UI local:       {local}")
    if public_url:
        print(f"  UI pública:     {public_url}")
    print(f"  API Key:        {cfg.api_key}")
    print(f"  OpenAI base:    {ui}/v1")
    print(f"  Chat endpoint:  {ui}/v1/chat/completions")
    print("-" * 70)
    print("  Abre la URL en tu navegador para cargar modelos (Ollama / llama.cpp / HuggingFace).")
    print("  Para agentes remotos, usa la URL pública + /v1 como base_url.")
    print("=" * 70 + "\n")


def run(cfg: Config):
    app = create_app(cfg)
    state = get_state()

    # Start tunnel in background so UI is available; update state when ready
    def _tunnel_worker():
        if cfg.tunnel == "none":
            return
        try:
            t, url = start_tunnel(cfg.tunnel, cfg.port, cfg.ngrok_token)
            state.tunnel_obj = t
            state.public_url = url
            print(f"[launcher] Túnel activo: {url}")
            _print_banner(cfg, url)
        except Exception as e:
            print(f"[launcher] No se pudo iniciar túnel ({cfg.tunnel}): {e}")
            print("[launcher] La UI sigue disponible localmente.")

    import threading
    if cfg.tunnel != "none":
        threading.Thread(target=_tunnel_worker, daemon=True).start()

    # Pre-download llama-server binary on startup so it's ready
    def _prefetch_llama():
        try:
            from . import backends
            state = get_state()
            def _on_progress(info):
                state.download_progress = info
                if info.get("active"):
                    pct = info.get("percent")
                    label = info.get("label", "llama.cpp")
                    state.status_detail = f"Preparando llama.cpp: {label} ({pct:.1f}%)" if pct is not None else f"Preparando llama.cpp: {label}"
            backends.set_progress_callback(_on_progress)
            state.status_detail = "Preparando llama.cpp..."
            path = backends.ensure_llama_server(cfg.llama_binary)
            print(f"[launcher] llama-server listo: {path}")
            if state.status == "stopped":
                state.status_detail = "llama.cpp listo. Selecciona un modelo para iniciar."
            state.download_progress = {"active": False, "percent": 100}
        except Exception as e:
            print(f"[launcher] No se pudo pre-descargar llama-server: {e}")
            try:
                state = get_state()
                if state.status == "stopped":
                    state.status_detail = f"Aviso: no se pudo preparar llama.cpp: {e}"
                state.download_progress = {"active": False}
            except Exception:
                pass
        finally:
            try:
                from . import backends
                backends.set_progress_callback(None)
            except Exception:
                pass

    # Auto-start model if provided via CLI
    if cfg.model:
        from .webapp import _start_backend_thread
        req_cfg = {
            "backend": cfg.backend,
            "model": cfg.model,
            "n_ctx": cfg.n_ctx,
            "n_gpu_layers": cfg.n_gpu_layers,
            "cache_type_k": cfg.cache_type_k,
            "cache_type_v": cfg.cache_type_v,
            "flash_attn": cfg.flash_attn,
            "n_batch": cfg.n_batch,
            "n_threads": cfg.n_threads,
            "jinja": cfg.jinja,
            "ctx_shift": cfg.ctx_shift,
            "keep": cfg.keep,
        }
        threading.Thread(target=_start_backend_thread, args=(get_state(), req_cfg), daemon=True).start()
        print(f"[launcher] Auto-iniciando modelo: {cfg.model}")
    else:
        # Prefetch in background only when not auto-starting a model
        if cfg.backend == "llama.cpp":
            threading.Thread(target=_prefetch_llama, daemon=True).start()

    _print_banner(cfg, None)

    # Graceful shutdown on signals
    def _sig(*_):
        print("\n[launcher] Saliendo...")
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _sig)
    try:
        signal.signal(signal.SIGTERM, _sig)
    except Exception:
        pass

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    run(cfg)
