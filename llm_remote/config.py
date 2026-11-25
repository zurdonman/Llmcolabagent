import argparse
import os
import secrets
from dataclasses import dataclass, field


@dataclass
class Config:
    backend: str = "llama.cpp"  # "llama.cpp" or "ollama"
    model: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    tunnel: str = "cloudflared"  # "cloudflared", "ngrok", "none"
    ngrok_token: str = ""
    # llama.cpp specific
    n_ctx: int = 4096
    n_gpu_layers: int = -1  # -1 = auto/all on GPU if available
    llama_binary: str = ""  # path to llama-server; auto-downloaded if empty
    cache_type_k: str = ""  # f32,f16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1
    cache_type_v: str = ""
    flash_attn: bool = False
    n_batch: int = 0  # 0 = llama-server default
    n_threads: int = 0  # 0 = auto
    ctx_shift: bool = True  # context shifting / sliding window to avoid OOM
    defrag_thold: float = -1.0  # -1 = default, 0-1 to enable defrag
    keep: int = 0  # tokens to keep when shifting (0 = auto)
    # ollama specific
    ollama_host: str = "127.0.0.1:11434"
    # misc
    extra_args: list = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def local_url(self) -> str:
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run an LLM locally/Colab and expose it as an OpenAI-compatible API with a web UI."
    )
    p.add_argument("--backend", choices=["llama.cpp", "ollama"], default=os.getenv("LLM_BACKEND", "llama.cpp"))
    p.add_argument("--model", default=os.getenv("LLM_MODEL", ""),
                   help="Model path/URL/GGUF for llama.cpp, or model name for ollama (e.g. qwen2.5:7b)")
    p.add_argument("--host", default=os.getenv("LLM_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("LLM_PORT", "8000")))
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""),
                   help="API key for the exposed endpoint. Generated if omitted.")
    p.add_argument("--tunnel", choices=["cloudflared", "ngrok", "none"], default=os.getenv("LLM_TUNNEL", "cloudflared"))
    p.add_argument("--ngrok-token", default=os.getenv("NGROK_AUTHTOKEN", ""))
    p.add_argument("--n-ctx", type=int, default=int(os.getenv("LLM_N_CTX", "4096")))
    p.add_argument("--n-gpu-layers", type=int, default=int(os.getenv("LLM_N_GPU_LAYERS", "-1")))
    p.add_argument("--llama-binary", default=os.getenv("LLAMA_SERVER_BIN", ""),
                   help="Path to llama-server binary. Auto-downloaded if empty.")
    p.add_argument("--cache-type-k", default=os.getenv("LLM_CACHE_TYPE_K", ""),
                   help="KV cache type for K (f16,q8_0,q4_0,...) llama.cpp only")
    p.add_argument("--cache-type-v", default=os.getenv("LLM_CACHE_TYPE_V", ""),
                   help="KV cache type for V (f16,q8_0,q4_0,...) llama.cpp only")
    p.add_argument("--flash-attn", action="store_true", default=os.getenv("LLM_FLASH_ATTN", "").lower() in ("1","true","yes"))
    p.add_argument("--n-batch", type=int, default=int(os.getenv("LLM_N_BATCH", "0")))
    p.add_argument("--n-threads", type=int, default=int(os.getenv("LLM_N_THREADS", "0")))
    p.add_argument("--ctx-shift", action="store_true", default=os.getenv("LLM_CTX_SHIFT", "1").lower() in ("1","true","yes"))
    p.add_argument("--no-ctx-shift", action="store_true", help="Disable context shifting")
    p.add_argument("--defrag-thold", type=float, default=float(os.getenv("LLM_DEFRAG_THOLD", "-1")))
    p.add_argument("--keep", type=int, default=int(os.getenv("LLM_KEEP", "0")))
    p.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "127.0.0.1:11434"))
    p.add_argument("--no-ui", action="store_true", help="Do not open browser UI automatically (UI still served)")
    p.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args passed to backend server after --")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    extra = list(args.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]
    cfg = Config(
        backend=args.backend,
        model=args.model,
        host=args.host,
        port=args.port,
        tunnel=args.tunnel,
        ngrok_token=args.ngrok_token,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        llama_binary=args.llama_binary,
        cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v,
        flash_attn=args.flash_attn,
        n_batch=args.n_batch,
        n_threads=args.n_threads,
        ctx_shift=(args.ctx_shift and not args.no_ctx_shift),
        defrag_thold=args.defrag_thold,
        keep=args.keep,
        ollama_host=args.ollama_host,
        extra_args=extra,
    )
    if args.api_key:
        cfg.api_key = args.api_key
    return cfg
