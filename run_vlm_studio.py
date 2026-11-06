"""
run_vlm_studio.py
Main entrypoint for Vision AI Dataset Labeling, Booru Tagging, Multimodal Chat & Smart File Manager Studio.
Key Architectural Guarantees:
1. Bootstraps llama.cpp CUDA 12.x (`llama-server` & `llama-mtmd-cli`) if not already configured.
2. Pre-registers all `/studio_ui` and `/studio_api/*` FastAPI routes so no initial request hits a 404 race condition.
3. Launches Gradio tunnel (`gradio.live`) embedding `/studio_ui` full-screen.
"""

import os
import sys
import time
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
from bootstrap_vlm import bootstrap_llama_cpp
from config.vlm_catalog import DEFAULT_MODEL_KEY, MODEL_REGISTRY
from core.llama_manager import ensure_llama_server
from core.tagger_manager import ensure_pixai_tagger
from ui.studio_routes import register_studio_routes


def create_gradio_shell() -> gr.Blocks:
    """Create the full-viewport Gradio Blocks container hosting `/studio_ui`."""
    custom_css = (
        "body, html, .gradio-container { margin:0 !important; padding:0 !important; "
        "max-width:100vw !important; overflow:hidden !important; background:#090b10 !important; } "
        "footer { display:none !important; }"
    )
    with gr.Blocks(
        title="Vision AI Labeling, PixAI Booru & Chat Studio",
        css=custom_css,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(
            '<iframe src="/studio_ui" '
            'style="position:fixed; inset:0; width:100vw; height:100vh; border:none; '
            'margin:0; padding:0; z-index:999999; background:#090b10;"></iframe>'
        )
    return demo


def main() -> None:
    # 1. Ensure llama.cpp CUDA binaries are installed & linked
    if not os.environ.get("VLM_LLAMA_SERVER") or not os.path.exists(os.environ.get("VLM_LLAMA_SERVER", "")):
        bootstrap_llama_cpp()

    default_model = os.environ.get("VLM_DEFAULT_MODEL", DEFAULT_MODEL_KEY)
    if default_model not in MODEL_REGISTRY:
        default_model = DEFAULT_MODEL_KEY

    prewarm = os.environ.get("VLM_PREWARM_MODEL", "1") == "1"
    if prewarm:
        print(f"[+] Pre-warming model default: {default_model} ...", flush=True)
        try:
            spec = MODEL_REGISTRY[default_model]
            if spec.get("engine") == "pixai_tagger":
                ensure_pixai_tagger()
            else:
                ensure_llama_server(default_model)
        except Exception as e:
            print(f"[!] Peringatan pre-warm model ({e}); model dapat dimuat langsung dari tombol UI.", flush=True)

    demo = create_gradio_shell()
    ui_port = int(os.environ.get("VLM_UI_PORT", "7865"))

    # Launch Gradio with prevent_thread_lock=True and immediately attach routes to the underlying FastAPI app
    app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=ui_port,
        share=True,
        show_error=True,
        prevent_thread_lock=True,
    )
    register_studio_routes(app)

    print(f"VLM_LOCAL_URL={local_url}", flush=True)
    print(f"VLM_SHARE_URL={share_url}", flush=True)

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
