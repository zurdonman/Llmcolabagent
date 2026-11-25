# LLM Remote (Plug & Play)

Aplicación web para ejecutar modelos LLM en **Google Colab** o en tu **PC** usando **Ollama** o **llama.cpp**, con panel visual para:

- Cargar modelos desde **Ollama** (lista automática de modelos locales + pull).
- Buscar y cargar modelos **GGUF desde HuggingFace** en tiempo real.
- Configurar memoria/rendimiento: `n_ctx`, `n_gpu_layers`, **KV cache K/V quantization** (`q8_0`, `q4_0`, etc.), Flash Attention, batch, threads.
- Exponer una **API OpenAI-compatible** (`/v1/chat/completions`) protegida con API key.
- Publicar la app por **Cloudflare Tunnel** para acceder desde otra red (móvil, otro PC, harness de agentes).

## 1 comando para lanzarla

```bash
git clone https://github.com/zurdonman/Llmcolabagent.git && cd Llmcolabagent && python run.py
```

`run.py` instala dependencias automáticamente si faltan (fastapi, uvicorn, httpx, pyngrok).

Al arrancar verás:
- **UI local**: `http://127.0.0.1:8000`
- **UI pública**: `https://....trycloudflare.com` (si el túnel está activo)
- **API Key**: para consumir la API remotamente

Abre la UI, elige backend/modelo, ajusta configuración y pulsa **Cargar / Iniciar modelo**.

## Google Colab

Usa `Colab_LLM_API.ipynb`. Ejecuta la celda de lanzamiento y abre la URL pública que aparece en la salida. Desde ahí cargas el modelo y obtienes el endpoint `/v1` para tu agente.

## Uso desde escritorio

```bash
git clone https://github.com/zurdonman/Llmcolabagent.git
cd Llmcolabagent
python run.py --tunnel cloudflared --port 8000
```

Opciones útiles:

```bash
python run.py --help
```

| Opción | Descripción | Default |
|---|---|---|
| `--backend` | `llama.cpp` u `ollama` | `llama.cpp` |
| `--model` | Modelo inicial (opcional, también se puede cargar desde UI) | vacío |
| `--host` | Host del servidor | `0.0.0.0` |
| `--port` | Puerto | `8000` |
| `--api-key` | API key fija | autogenerada |
| `--tunnel` | `cloudflared`, `ngrok`, `none` | `cloudflared` |
| `--ngrok-token` | Token ngrok | vacío |
| `--n-ctx` | Contexto llama.cpp | `4096` |
| `--n-gpu-layers` | Capas GPU (`-1` auto) | `-1` |
| `--cache-type-k` | KV cache K: `f16,q8_0,q4_0,...` | auto |
| `--cache-type-v` | KV cache V: `f16,q8_0,q4_0,...` | auto |
| `--flash-attn` | Activa Flash Attention | off |
| `--n-batch` | Batch size | auto |
| `--n-threads` | Threads CPU | auto |

## Panel web (qué puedes hacer)

### Ollama
- Ver modelos instalados localmente en Ollama.
- Hacer `pull` de un modelo nuevo por nombre (ej. `qwen2.5:1.5b`).
- Cargarlo y exponerlo por API.

### llama.cpp + HuggingFace
- Buscar modelos GGUF en HuggingFace en tiempo real.
- Ver archivos `.gguf` del repo seleccionado.
- Seleccionar cuantización/archivo y cargarlo.
- También acepta ruta local `.gguf` o URL directa.

### Configuración de memoria/rendimiento
- **KV cache K/V**: cuantiza la cache KV para reducir VRAM/RAM en contextos largos. Útil en Colab.
- **Flash Attention**: `-fa` para mejor rendimiento/memoria en GPUs compatibles.
- **n_ctx, n_gpu_layers, n_batch, n_threads**: control de carga del modelo.

## Consumir la API (móvil / agente)

La app expone un endpoint OpenAI-compatible. Usa la URL pública (o local) como `base_url` terminada en `/v1`.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://TU-URL.trycloudflare.com/v1",
    api_key="TU_API_KEY",
)

resp = client.chat.completions.create(
    model="qwen2.5:1.5b",  # o el modelo cargado
    messages=[{"role": "user", "content": "Hola desde mi agente"}],
)
print(resp.choices[0].message.content)
```

### curl

```bash
curl https://TU-URL.trycloudflare.com/v1/chat/completions \
  -H "Authorization: Bearer TU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hola"}]}'
```

## Estructura

```text
.
├── Colab_LLM_API.ipynb      # Lanzador Colab
├── run.py                   # Entrada plug-and-play
├── requirements.txt
├── llm_remote/
│   ├── webapp.py            # FastAPI: UI + API management + proxy /v1
│   ├── ui.html              # Panel web
│   ├── backends.py          # llama.cpp / Ollama
│   ├── hf.py                # Búsqueda HuggingFace
│   ├── tunnel.py            # Cloudflare/ngrok
│   ├── launcher.py          # Arranque
│   └── config.py
└── README.md
```

## Notas de seguridad

- La API `/v1/*` requiere `Authorization: Bearer <API_KEY>`.
- El túnel Cloudflare quick es temporal (cambia al reiniciar).
- No compartas tu API key públicamente.

## Troubleshooting

- **No veo modelos en Ollama**: pulsa "Actualizar" o instala/inicia Ollama; el backend puede instalarse en Linux, en Windows/Mac instálalo desde ollama.com.
- **Falla carga GGUF**: revisa `logs/backend.log`; prueba una cuantización menor o reduce `n_ctx` / usa KV `q8_0`.
- **Sin URL pública**: revisa conectividad; usa `--tunnel ngrok --ngrok-token ...` o `--tunnel none` para local.
- **Agente no conecta**: usa `base_url` con `/v1` al final y la API key correcta.
