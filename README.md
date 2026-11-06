# ⚡ Vision AI Dataset Labeling, PixAI Booru Tagger, Multimodal Chat & Smart File Manager Studio (T4 CUDA Edition)

Sistem produksi modular untuk **Auto-Captioning Dataset LoRA/Fine-Tuning (Flux, Qwen-Image 2.1, MOSS, Chroma1-HD, Anima, Krea 2, Wan 2.1)**, **Ultra-Fast Anime Booru Tagging (`pixai-labs/pixai-tagger-v0.9`)**, **Multimodal Vision Chat**, dan **Smart Dataset/VRAM Manager** di atas GPU NVIDIA Tesla T4 (15 GiB VRAM).

## Struktur Modul

```text
vlm-caption-studio/
├── requirements.txt                  # Dependensi terverifikasi (Gradio 5, FastAPI, timm, CUDA 12 libs)
├── bootstrap_vlm.py                  # Smart installer binary llama.cpp CUDA 12.x (llama-server & llama-mtmd-cli)
├── run_vlm_studio.py                 # Entrypoint utama FastAPI + Gradio Tunnel
├── config/
│   ├── __init__.py
│   └── vlm_catalog.py                # Katalog 7 VLM GGUF + PixAI Tagger v0.9 + Per-Model Routing Adapters + 8 Style Presets
├── core/
│   ├── __init__.py
│   ├── llama_manager.py              # Manajer llama-server (-c 8192, -np 1, KV-cache q8_0, anti-truncation headroom, VRAM/Disk telemetry)
│   ├── tagger_manager.py             # PyTorch + timm EVA02-Large runner untuk PixAI Tagger v0.9 (Standalone & Hybrid VLM Assistant)
│   ├── dataset_manager.py            # Natural sort, Safe ZIP extraction anti-__MACOSX, Batch Delete, Bulk Trigger Edit, Google Drive Sync
│   └── text_cleaner.py               # Anti-leak <think> cleaner, unfinished sentence repair, & Dual Word/Token estimator
├── ui/
│   ├── __init__.py
│   ├── studio_routes.py              # Non-blocking FastAPI routes (asyncio.to_thread)
│   └── templates/
│       └── studio.html               # Single-Page Application (Pagination, Lightbox Split Editor, Live Timers, Persistent Vision Chat)
└── notebooks/
    └── VLM_Caption_Studio_Colab.ipynb # Notebook Google Colab Tipis (2 Sel: Git Clone/Branch Setup & Launch)
```
