import asyncio
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from . import backends, hf
from .config import Config
from .tunnel import start_tunnel


class AppState:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.backend_proc = None  # subprocess.Popen or None
        self.backend_type: Optional[str] = None  # llama.cpp / ollama
        self.upstream_base: Optional[str] = None
        self.model_loaded: str = ""
        self.status: str = "stopped"  # stopped|starting|running|error
        self.status_detail: str = ""
        self.tunnel_obj = None
        self.public_url: Optional[str] = None
        self.log_path = Path(__file__).resolve().parent.parent / "logs" / "backend.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = None
        self.download_progress: dict = {"active": False}

    def open_log(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)
        return self._log_file

    def close_log(self):
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None


STATE: Optional[AppState] = None


def get_state() -> AppState:
    assert STATE is not None
    return STATE


def _wait_for_http(url: str, timeout: float = 300.0, proc=None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"Backend terminó con código {proc.returncode}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "llm-remote/1.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                if 200 <= r.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _stop_backend(state: AppState):
    backends.stop_process(state.backend_proc)
    state.backend_proc = None
    state.backend_type = None
    state.upstream_base = None
    state.model_loaded = ""
    state.status = "stopped"
    state.status_detail = ""
    state.download_progress = {"active": False}
    state.close_log()


def _start_backend_thread(state: AppState, req_cfg: dict):
    """Start backend in background thread, updating state."""
    with state.lock:
        if state.status == "starting":
            return
        # stop existing
        _stop_backend(state)
        state.status = "starting"
        state.status_detail = "Iniciando backend..."
        state.download_progress = {"active": False}

    def _on_progress(info):
        state.download_progress = info
        if info.get("active"):
            pct = info.get("percent")
            label = info.get("label", "descarga")
            if pct is not None:
                state.status_detail = f"Descargando: {label} ({pct:.1f}%)"
            else:
                state.status_detail = f"Descargando: {label}"

    backends.set_progress_callback(_on_progress)
    try:
        cfg = state.cfg
        # apply overrides from request
        cfg.backend = req_cfg.get("backend", cfg.backend)
        cfg.model = req_cfg.get("model", cfg.model)
        cfg.n_ctx = int(req_cfg.get("n_ctx", cfg.n_ctx))
        cfg.n_gpu_layers = int(req_cfg.get("n_gpu_layers", cfg.n_gpu_layers))
        cfg.cache_type_k = str(req_cfg.get("cache_type_k", cfg.cache_type_k) or "")
        cfg.cache_type_v = str(req_cfg.get("cache_type_v", cfg.cache_type_v) or "")
        cfg.flash_attn = bool(req_cfg.get("flash_attn", cfg.flash_attn))
        cfg.n_batch = int(req_cfg.get("n_batch", cfg.n_batch))
        cfg.n_threads = int(req_cfg.get("n_threads", cfg.n_threads))
        cfg.n_parallel = int(req_cfg.get("n_parallel", cfg.n_parallel))
        cfg.jinja = bool(req_cfg.get("jinja", cfg.jinja))
        cfg.ctx_shift = bool(req_cfg.get("ctx_shift", cfg.ctx_shift))
        cfg.keep = int(req_cfg.get("keep", cfg.keep))
        if req_cfg.get("api_key"):
            cfg.api_key = str(req_cfg["api_key"])

        if not cfg.model:
            raise ValueError("Debes especificar un modelo")

        log_file = state.open_log()
        internal_port = cfg.port + 1 if cfg.backend == "llama.cpp" else 11434
        state.status_detail = f"Arrancando {cfg.backend}..."
        proc, backend_type, upstream_base = backends.start_backend(cfg, internal_port, log_file)

        with state.lock:
            state.backend_proc = proc
            state.backend_type = backend_type
            state.upstream_base = upstream_base

        health_url = upstream_base + ("/health" if backend_type == "llama.cpp" else "/api/tags")
        state.status_detail = "Esperando a que el modelo esté listo..."
        ok = _wait_for_http(health_url, timeout=600, proc=proc)
        if not ok:
            raise RuntimeError("El backend no respondió a tiempo. Revisa logs/backend.log")

        with state.lock:
            state.status = "running"
            state.model_loaded = cfg.model
            state.status_detail = "Modelo listo"
            state.download_progress = {"active": False, "percent": 100}
    except Exception as e:
        with state.lock:
            backends.stop_process(state.backend_proc)
            state.backend_proc = None
            state.backend_type = None
            state.upstream_base = None
            state.model_loaded = ""
            state.status = "error"
            state.status_detail = str(e)
            state.download_progress = {"active": False}
            state.close_log()
    finally:
        backends.set_progress_callback(None)


def create_app(cfg: Config) -> FastAPI:
    global STATE
    STATE = AppState(cfg)
    state = STATE

    app = FastAPI(title="LLM Remote", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Proxy client is recreated as needed; for simplicity create one lazily per upstream
    proxy_client: dict[str, httpx.AsyncClient] = {}

    async def get_client() -> httpx.AsyncClient:
        upstream = state.upstream_base
        if not upstream:
            raise HTTPException(status_code=503, detail="No hay modelo cargado. Inicia uno desde la UI.")
        c = proxy_client.get(upstream)
        if c is None or c.is_closed:
            c = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(300.0, connect=10.0))
            proxy_client[upstream] = c
        return c

    @app.on_event("shutdown")
    async def _shutdown():
        for c in proxy_client.values():
            try:
                await c.aclose()
            except Exception:
                pass
        _stop_backend(state)
        t = state.tunnel_obj
        if t is not None:
            try:
                if hasattr(t, "terminate"):
                    t.terminate()
                else:
                    from pyngrok import ngrok
                    ngrok.kill()
            except Exception:
                pass

    UI_PATH = Path(__file__).resolve().parent / "ui.html"

    @app.get("/")
    async def index():
        if UI_PATH.exists():
            return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>LLM Remote</h1><p>ui.html no encontrado.</p>")

    # ---------- Management API ----------

    @app.get("/api/status")
    async def api_status():
        dp = dict(state.download_progress) if isinstance(state.download_progress, dict) else {"active": False}
        # If no active own-download, try to infer progress from backend log (llama-server HF download etc.)
        if not dp.get("active") and state.status == "starting":
            try:
                if state.log_path.exists():
                    tail = state.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                    import re as _re
                    for line in reversed(tail):
                        m = _re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", line)
                        if m:
                            pct = float(m.group(1))
                            if 0 <= pct <= 100:
                                dp = {"active": True, "label": "modelo/backend (ver log)", "percent": pct}
                                break
            except Exception:
                pass
        return {
            "status": state.status,
            "status_detail": state.status_detail,
            "backend": state.cfg.backend,
            "model": state.model_loaded or state.cfg.model,
            "model_loaded": state.model_loaded,
            "public_url": state.public_url,
            "local_url": f"http://127.0.0.1:{state.cfg.port}",
            "api_key": state.cfg.api_key,
            "port": state.cfg.port,
            "download_progress": dp,
            "config": {
                "n_ctx": state.cfg.n_ctx,
                "n_gpu_layers": state.cfg.n_gpu_layers,
                "cache_type_k": state.cfg.cache_type_k,
                "cache_type_v": state.cfg.cache_type_v,
                "flash_attn": state.cfg.flash_attn,
                "n_batch": state.cfg.n_batch,
                "n_threads": state.cfg.n_threads,
                "n_parallel": state.cfg.n_parallel,
                "jinja": state.cfg.jinja,
                "ctx_shift": state.cfg.ctx_shift,
                "keep": state.cfg.keep,
            },
        }

    @app.get("/api/ollama/models")
    async def api_ollama_models():
        models = backends.list_ollama_models(state.cfg.ollama_host)
        return {"running": backends.ollama_running(state.cfg.ollama_host), "models": models}

    @app.get("/api/ollama/gguf")
    async def api_ollama_gguf():
        models = await asyncio.to_thread(backends.list_ollama_gguf_models)
        return {"models": models}

    @app.post("/api/ollama/pull")
    async def api_ollama_pull(req: Request):
        body = await req.json()
        model = (body.get("model") or "").strip()
        if not model:
            raise HTTPException(status_code=400, detail="model requerido")
        try:
            # run in thread to not block event loop too badly? subprocess.run blocks anyway; use thread
            await asyncio.to_thread(backends.ollama_pull, model)
            return {"ok": True, "model": model}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/hf/search")
    async def api_hf_search(q: str = "", limit: int = 20):
        try:
            results = await asyncio.to_thread(hf.search_models, q, limit)
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/hf/files")
    async def api_hf_files(repo: str):
        try:
            files = await asyncio.to_thread(hf.list_gguf_files, repo)
            return {"repo": repo, "files": files}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/server/start")
    async def api_server_start(req: Request):
        body = await req.json() if req.headers.get("content-type","").startswith("application/json") else {}
        with state.lock:
            if state.status == "starting":
                return {"ok": True, "status": state.status, "detail": state.status_detail}
        t = threading.Thread(target=_start_backend_thread, args=(state, body), daemon=True)
        t.start()
        return {"ok": True, "status": "starting"}

    @app.post("/api/server/stop")
    async def api_server_stop():
        with state.lock:
            _stop_backend(state)
        return {"ok": True, "status": "stopped"}

    @app.get("/api/logs")
    async def api_logs(tail: int = 200):
        p = state.log_path
        if not p.exists():
            return {"lines": []}
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()[-max(1, min(tail, 2000)):]
            return {"lines": lines}
        except Exception as e:
            return {"lines": [f"error leyendo log: {e}"]}

    @app.get("/api/logs/download")
    async def api_logs_download():
        parts = []
        app_log = Path(__file__).resolve().parent.parent / "logs" / "app.log"
        backend_log = state.log_path
        for label, p in [("APP LOG", app_log), ("BACKEND LOG", backend_log)]:
            parts.append(f"===== {label}: {p} =====")
            if p.exists():
                try:
                    parts.append(p.read_text(encoding="utf-8", errors="replace"))
                except Exception as e:
                    parts.append(f"[error leyendo {p}: {e}]")
            else:
                parts.append("(no existe)")
            parts.append("")
        content = "\n".join(parts)
        headers = {"Content-Disposition": "attachment; filename=llm-remote-logs.txt"}
        return Response(content=content, media_type="text/plain; charset=utf-8", headers=headers)

    # ---------- OpenAI-compatible proxy (/v1/...) ----------

    def _check_auth(request: Request):
        api_key = state.cfg.api_key
        if not api_key:
            return
        auth = request.headers.get("authorization") or ""
        x_api = request.headers.get("x-api-key") or ""
        ok = False
        if auth.lower().startswith("bearer ") and auth.split(" ", 1)[1].strip() == api_key:
            ok = True
        if x_api == api_key:
            ok = True
        if request.query_params.get("key") == api_key:
            ok = True
        if not ok:
            raise HTTPException(status_code=401, detail="Unauthorized. Usa Authorization: Bearer <API_KEY>")

    @app.get("/v1/models")
    async def v1_models(request: Request):
        _check_auth(request)
        if state.status != "running" or not state.upstream_base:
            # Return a friendly empty-ish response
            return {"object": "list", "data": [{"id": state.cfg.model or "none", "object": "model"}]}
        client = await get_client()
        try:
            r = await client.get("/v1/models")
            return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
        except Exception:
            return {"object": "list", "data": [{"id": state.model_loaded or state.cfg.model, "object": "model"}]}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def v1_proxy(path: str, request: Request):
        if request.method == "OPTIONS":
            return Response(status_code=204)
        _check_auth(request)
        if state.status != "running" or not state.upstream_base:
            raise HTTPException(status_code=503, detail="No hay modelo cargado. Inicia uno desde la UI.")

        client = await get_client()
        url_path = "/v1/" + path
        if request.url.query:
            url_path += "?" + request.url.query

        headers = dict(request.headers)
        for h in ["host", "content-length", "transfer-encoding", "connection"]:
            headers.pop(h, None)
            headers.pop(h.title(), None)

        body = await request.body()
        is_stream = False
        if body and "application/json" in request.headers.get("content-type", ""):
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    if payload.get("stream") is True:
                        is_stream = True
                    if state.model_loaded and (not payload.get("model") or payload.get("model") in ("gpt-3.5-turbo","gpt-4","default")):
                        payload["model"] = state.model_loaded
                        body = json.dumps(payload).encode("utf-8")
            except Exception:
                pass

        req = client.build_request(request.method, url_path, headers=headers, content=body)
        if is_stream:
            upstream_resp = await client.send(req, stream=True)
            async def gen():
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        yield chunk
                finally:
                    await upstream_resp.aclose()
            resp_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in ("content-length","transfer-encoding","connection")}
            return StreamingResponse(gen(), status_code=upstream_resp.status_code, headers=resp_headers, media_type=upstream_resp.headers.get("content-type"))
        upstream_resp = await client.send(req, stream=False)
        resp_headers = {k: v for k, v in upstream_resp.headers.items() if k.lower() not in ("content-length","transfer-encoding","connection","content-encoding")}
        return Response(content=upstream_resp.content, status_code=upstream_resp.status_code, headers=resp_headers, media_type=upstream_resp.headers.get("content-type"))

    return app
