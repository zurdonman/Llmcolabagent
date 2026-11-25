"""HuggingFace model search helpers (GGUF-focused)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any


UA = {"User-Agent": "llm-remote/1.0"}


def _get_json(url: str, timeout: int = 20) -> Any:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def search_models(query: str, limit: int = 20) -> list[dict]:
    """Search HuggingFace models, biased to GGUF repos."""
    q = (query or "").strip()
    params = {
        "search": q,
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(max(1, min(limit, 50))),
        "full": "false",
    }
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(params)
    data = _get_json(url)
    results = []
    for m in data if isinstance(data, list) else []:
        results.append({
            "id": m.get("id") or m.get("modelId"),
            "downloads": m.get("downloads"),
            "likes": m.get("likes"),
            "lastModified": m.get("lastModified"),
            "pipeline_tag": m.get("pipeline_tag"),
            "tags": m.get("tags", [])[:12],
        })
    return results


def list_gguf_files(repo_id: str) -> list[dict]:
    """List GGUF files in a HF repo (from siblings)."""
    repo_id = repo_id.strip().strip("/")
    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id inválido")
    url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id, safe='/')}?blobs=true"
    data = _get_json(url)
    siblings = data.get("siblings", []) if isinstance(data, dict) else []
    files = []
    for s in siblings:
        name = s.get("rfilename") or s.get("filename") or ""
        if not name.lower().endswith(".gguf"):
            continue
        size = s.get("size")
        files.append({
            "filename": name,
            "size": size,
            "size_mb": round(size / (1024 * 1024), 1) if isinstance(size, (int, float)) else None,
            "download_url": f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(name)}",
        })
    # Sort by size desc, then name
    files.sort(key=lambda x: (x.get("size") or 0, x.get("filename") or ""))
    return files
