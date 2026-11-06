"""
ui/studio_routes.py
100% Non-Blocking FastAPI Endpoints for Vision AI Dataset Labeling, Booru Tagging,
Multimodal Chat & Smart File Manager Studio.
All heavy I/O and GPU operations are dispatched via `asyncio.to_thread(...)` so UI telemetry,
image thumbnails, and caption saving respond in <10ms even during heavy batch inference.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from config.vlm_catalog import (
    MODEL_REGISTRY,
    STYLE_PRESETS,
    DEFAULT_MODEL_KEY,
    PIXAI_TAGGER_KEY,
    get_model_spec,
)
from core.llama_manager import (
    ACTIVE_STATE,
    ensure_llama_server,
    unload_vlm_server,
    run_vlm_caption_single,
    run_multimodal_chat,
    get_system_telemetry,
    delete_downloaded_model,
    image_to_data_uri,
)
from core.tagger_manager import (
    ensure_pixai_tagger,
    unload_pixai_tagger,
    run_pixai_tagger,
)
from core.dataset_manager import (
    DEFAULT_DATASET_DIR,
    scan_dataset_items,
    save_single_caption,
    delete_single_item,
    delete_batch_items,
    process_uploaded_file_bytes,
    create_dataset_zip,
    run_bulk_caption_edit,
    run_smart_file_manager,
)
from core.text_cleaner import estimate_token_count

TEMPLATE_PATH = Path(__file__).parent / "templates" / "studio.html"


def get_initial_boot_payload(folder: str = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    """Build initial JSON payload injected directly into studio.html for 0ms boot."""
    default_key = os.environ.get("VLM_DEFAULT_MODEL", DEFAULT_MODEL_KEY)
    if default_key not in MODEL_REGISTRY:
        default_key = DEFAULT_MODEL_KEY

    models_list = []
    for k, v in MODEL_REGISTRY.items():
        models_list.append({
            "key": k,
            "engine": v.get("engine", "llama_cpp"),
            "family": v.get("family", ""),
            "desc": v.get("desc", ""),
            "approx_size_gb": v.get("approx_size_gb", 5.8),
            "sampling": v.get("caption_sampling", {}),
        })

    return {
        "folder": folder,
        "models": models_list,
        "default_model": ACTIVE_STATE.get("model_key") or default_key,
        "pixai_tagger_key": PIXAI_TAGGER_KEY,
        "styles": STYLE_PRESETS,
    }


def build_studio_html(folder: str = DEFAULT_DATASET_DIR) -> str:
    """Read `ui/templates/studio.html` and inject `INITIAL_BOOT_JSON`."""
    raw_html = TEMPLATE_PATH.read_text(encoding="utf-8")
    boot_json = json.dumps(get_initial_boot_payload(folder))
    return raw_html.replace("__INITIAL_BOOT_JSON__", boot_json)


def register_studio_routes(app: FastAPI) -> None:
    """Register all `/studio_ui` and `/studio_api/*` routes on the FastAPI `app` instance."""

    @app.get("/studio_ui", response_class=HTMLResponse)
    async def route_studio_ui(folder: str = DEFAULT_DATASET_DIR):
        html_content = await asyncio.to_thread(build_studio_html, folder)
        return HTMLResponse(html_content)

    @app.get("/studio_api/init")
    async def route_init(folder: str = DEFAULT_DATASET_DIR):
        items, telemetry = await asyncio.gather(
            asyncio.to_thread(scan_dataset_items, folder),
            asyncio.to_thread(get_system_telemetry),
        )
        payload = get_initial_boot_payload(folder)
        payload["items"] = items
        payload["telemetry"] = telemetry
        return payload

    @app.get("/studio_api/telemetry")
    async def route_telemetry():
        data = await asyncio.to_thread(get_system_telemetry)
        return data

    @app.get("/studio_api/image")
    async def route_image(path: str, as_data_uri: int = 0):
        p = Path(path)
        if not p.is_file():
            return JSONResponse({"error": "Image file not found"}, status_code=404)
        if as_data_uri == 1:
            uri = await asyncio.to_thread(image_to_data_uri, str(p))
            return {"ok": True, "name": p.name, "data_uri": uri}
        return FileResponse(str(p))

    @app.post("/studio_api/save_caption")
    async def route_save_caption(req: Request):
        data = await req.json()
        img_path = data.get("img_path", "")
        caption = data.get("caption", "")
        res = await asyncio.to_thread(save_single_caption, img_path, caption)
        return res

    @app.post("/studio_api/delete_item")
    async def route_delete_item(req: Request):
        data = await req.json()
        img_path = data.get("img_path", "")
        folder = data.get("folder", DEFAULT_DATASET_DIR)
        res = await asyncio.to_thread(delete_single_item, img_path)
        items = await asyncio.to_thread(scan_dataset_items, folder)
        res["items"] = items
        return res

    @app.post("/studio_api/delete_batch")
    async def route_delete_batch(req: Request):
        data = await req.json()
        folder = data.get("folder", DEFAULT_DATASET_DIR)
        selected_paths = data.get("selected_img_paths")
        delete_all = bool(data.get("delete_all", False))
        res = await asyncio.to_thread(
            delete_batch_items,
            folder_path=folder,
            selected_img_paths=selected_paths,
            delete_all=delete_all,
        )
        return res

    @app.post("/studio_api/upload")
    async def route_upload(folder: str = Form(DEFAULT_DATASET_DIR), files: list[UploadFile] = File(...)):
        added_total = 0
        corrupt_total = 0
        for uf in files:
            fname = uf.filename or "upload.png"
            content = await uf.read()
            added, corrupt = await asyncio.to_thread(process_uploaded_file_bytes, folder, fname, content)
            added_total += added
            corrupt_total += corrupt
        items = await asyncio.to_thread(scan_dataset_items, folder)
        return {
            "ok": True,
            "added": added_total,
            "corrupt": corrupt_total,
            "items": items,
        }

    @app.post("/studio_api/switch_model")
    async def route_switch_model(req: Request):
        data = await req.json()
        model_key = data.get("model_key", DEFAULT_MODEL_KEY)
        spec = get_model_spec(model_key)
        try:
            if spec.get("engine") == "pixai_tagger":
                await asyncio.to_thread(ensure_pixai_tagger)
            else:
                await asyncio.to_thread(ensure_llama_server, model_key)
            telemetry = await asyncio.to_thread(get_system_telemetry)
            return {
                "ok": True,
                "model_key": model_key,
                "msg": f"Model '{model_key.split('[')[0].strip()}' siap di VRAM GPU.",
                "telemetry": telemetry,
            }
        except Exception as e:
            telemetry = await asyncio.to_thread(get_system_telemetry)
            return JSONResponse(
                {"ok": False, "msg": f"Gagal memuat model: {e}", "telemetry": telemetry},
                status_code=500,
            )

    @app.post("/studio_api/unload_vram")
    async def route_unload_vram(req: Request):
        data = await req.json()
        target = data.get("target", "all")
        if target == "tagger":
            await asyncio.to_thread(unload_pixai_tagger)
            msg = "PixAI Tagger v0.9 berhasil di-unload dari VRAM."
        elif target == "vlm":
            await asyncio.to_thread(unload_vlm_server, False)
            msg = "Model VLM (llama-server) berhasil di-unload dari VRAM."
        else:
            await asyncio.to_thread(unload_vlm_server, True)
            msg = "Seluruh model VLM & PixAI Tagger berhasil di-unload dari VRAM."
        telemetry = await asyncio.to_thread(get_system_telemetry)
        return {"ok": True, "msg": msg, "telemetry": telemetry}

    @app.post("/studio_api/delete_model_cache")
    async def route_delete_model_cache(req: Request):
        data = await req.json()
        model_key = data.get("model_key", "")
        res = await asyncio.to_thread(delete_downloaded_model, model_key)
        telemetry = await asyncio.to_thread(get_system_telemetry)
        res["telemetry"] = telemetry
        return res

    @app.post("/studio_api/generate_one")
    async def route_generate_one(req: Request):
        data = await req.json()
        img_path = data["img_path"]
        model_key = data.get("model_key", DEFAULT_MODEL_KEY)
        sys_prompt = data.get("sys_prompt", "")
        usr_prompt = data.get("usr_prompt", "Describe this image in detail.")
        max_tokens = int(data.get("max_tokens", 220))
        temperature = float(data.get("temperature", 0.2))
        subject_hint = data.get("subject_hint", "")
        hybrid_tagger_enabled = bool(data.get("hybrid_tagger_enabled", False))
        hybrid_mode = data.get("hybrid_mode", "grounding")
        general_threshold = float(data.get("general_threshold", 0.30))
        character_threshold = float(data.get("character_threshold", 0.75))
        replace_underscore = bool(data.get("replace_underscore", False))
        escape_parens = bool(data.get("escape_parens", False))
        exclude_tags = data.get("exclude_tags", "")

        res = await asyncio.to_thread(
            run_vlm_caption_single,
            img_path=img_path,
            model_key=model_key,
            sys_prompt=sys_prompt,
            usr_prompt=usr_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            subject_hint=subject_hint,
            hybrid_tagger_enabled=hybrid_tagger_enabled,
            hybrid_mode=hybrid_mode,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            replace_underscore=replace_underscore,
            escape_parens=escape_parens,
            exclude_tags=exclude_tags,
        )
        if res.get("ok") and res.get("caption"):
            await asyncio.to_thread(save_single_caption, img_path, res["caption"])
        return res

    @app.post("/studio_api/tag_one")
    async def route_tag_one(req: Request):
        """Run standalone PixAI Tagger v0.9 on a single image and optionally save to .txt."""
        data = await req.json()
        img_path = data["img_path"]
        save_to_file = bool(data.get("save_to_file", True))
        general_threshold = float(data.get("general_threshold", 0.30))
        character_threshold = float(data.get("character_threshold", 0.75))
        replace_underscore = bool(data.get("replace_underscore", False))
        escape_parens = bool(data.get("escape_parens", False))
        exclude_tags = data.get("exclude_tags", "")

        tag_res = await asyncio.to_thread(
            run_pixai_tagger,
            img_path=img_path,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            include_ip=True,
            replace_underscore=replace_underscore,
            escape_parens=escape_parens,
            exclude_tags=exclude_tags,
        )
        cap = tag_res.get("caption", "")
        stats = estimate_token_count(cap)
        if save_to_file and cap:
            await asyncio.to_thread(save_single_caption, img_path, cap)
        return {
            "ok": bool(cap),
            "caption": cap,
            "words": stats["words"],
            "tokens": stats["tokens"],
            "tagger_meta": tag_res,
        }

    @app.get("/studio_api/download_zip")
    async def route_download_zip(folder: str = DEFAULT_DATASET_DIR):
        out_zip = await asyncio.to_thread(create_dataset_zip, folder)
        return FileResponse(str(out_zip), filename=out_zip.name, media_type="application/zip")

    @app.post("/studio_api/bulk_edit")
    async def route_bulk_edit(req: Request):
        d = await req.json()
        res = await asyncio.to_thread(
            run_bulk_caption_edit,
            folder_path=d.get("folder", DEFAULT_DATASET_DIR),
            action_type=d.get("action_type", "apply"),
            trigger=d.get("trigger", ""),
            pos=d.get("pos", "front"),
            find_str=d.get("find", ""),
            replace_str=d.get("replace", ""),
            lowercase=bool(d.get("lowercase", False)),
            skip_dup=bool(d.get("skip_dup", True)),
            clean_commas=bool(d.get("clean_commas", True)),
            backup=bool(d.get("backup", True)),
            selected_img_paths=d.get("selected_img_paths"),
        )
        return res

    @app.post("/studio_api/chat")
    async def route_chat(req: Request):
        d = await req.json()
        res = await asyncio.to_thread(
            run_multimodal_chat,
            model_key=d.get("model_key", DEFAULT_MODEL_KEY),
            sys_prompt=d.get("sys_prompt", "You are a helpful, uncensored multimodal AI assistant."),
            history=d.get("history", []),
            user_message=d.get("message", ""),
            image_data_uri=d.get("image_data_uri"),
            temperature=float(d.get("temperature", 0.7)),
            max_tokens=int(d.get("max_tokens", 768)),
            enable_thinking=bool(d.get("enable_thinking", False)),
        )
        return res

    @app.post("/studio_api/manager")
    async def route_manager(req: Request):
        d = await req.json()
        res = await asyncio.to_thread(
            run_smart_file_manager,
            action=d.get("action", "Count Statistics"),
            source_path=d.get("source_path", DEFAULT_DATASET_DIR),
            destination_path=d.get("destination_path", "/content/dataset.zip"),
            rename_prefix=d.get("rename_prefix", "dataset_img"),
            active_dataset_dir=d.get("active_folder", DEFAULT_DATASET_DIR),
        )
        return res
