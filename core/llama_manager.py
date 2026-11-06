"""
core/llama_manager.py
Production Process & Memory Manager for llama-server & llama-mtmd-cli on NVIDIA Tesla T4 (15 GiB VRAM).
Features:
- Thread-safe model loading (`threading.Lock()`)
- Optimal T4 server flags (`-ngl 99 -c 8192 -np 1 -cb -ub 512 -b 1024 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --jinja`)
- Per-Model Prompt Routing (`system_and_user`, `joycaption_mode`, `user_merged`)
- Dynamic Qwen3.5 thinking toggle per API call
- Hybrid PixAI Tagger v0.9 integration (Grounding Prompt, Prepend Tags, Append Tags)
- Global Subject / Concept Hint injection
- Automatic server restart & `llama-mtmd-cli` fallback
- Full GPU VRAM / System RAM / Disk Cache Telemetry & GGUF Cleaner
"""

import os
import io
import re
import time
import glob
import shutil
import signal
import base64
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

try:
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError:
    hf_hub_download = None
    list_repo_files = None

from config.vlm_catalog import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    PIXAI_TAGGER_KEY,
    get_model_spec,
)
from core.text_cleaner import clean_vlm_response, extract_message_text, estimate_token_count
from core.tagger_manager import (
    TAGGER_STATE,
    TAGGER_CACHE_DIR,
    is_tagger_downloaded,
    run_pixai_tagger,
    build_hybrid_vlm_context,
    unload_pixai_tagger,
)

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

MODELS_CACHE_DIR = Path(os.environ.get("VLM_MODELS_DIR", "/content/vlm_models"))
SERVER_PORT = int(os.environ.get("VLM_SERVER_PORT", "8085"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
SERVER_PID_FILE = Path("/content/vlm_llama_server.pid")
SERVER_LOG_FILE = Path("/content/vlm_llama_server.log")

_SERVER_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

ACTIVE_STATE: Dict[str, Any] = {
    "status": "STANDBY",  # STANDBY | LOADING | ONLINE | ERROR
    "model_key": None,
    "model_path": None,
    "mmproj_path": None,
    "proc": None,
    "last_error": None,
}


def get_llama_binaries() -> Tuple[str, str, str]:
    """Return resolved paths for llama-server, llama-mtmd-cli, and llama-cli."""
    srv = os.environ.get("VLM_LLAMA_SERVER", "/content/llama-bin/cuda-12.8/llama-server")
    mtmd = os.environ.get("VLM_LLAMA_MTMD_CLI", "")
    cli = os.environ.get("VLM_LLAMA_CLI", "/content/llama-bin/cuda-12.8/llama-cli")
    if not mtmd:
        found = glob.glob("/content/llama-bin/**/llama-mtmd-cli", recursive=True)
        mtmd = found[0] if found else cli
    return srv, mtmd, cli


def _repo_local_dir(repo_id: str) -> Path:
    return MODELS_CACHE_DIR / repo_id.replace("/", "__")


def resolve_and_download_hf(repo_id: str, preferred_filename: str, fallback_glob: str = "*") -> str:
    """
    Download GGUF / MMProj file into MODELS_CACHE_DIR/<repo_id> so disk cache management
    is predictable, inspectable, and cleanable from Tab 3. Includes smart glob matching & wget fallback.
    """
    local_dir = _repo_local_dir(repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)

    direct_candidate = local_dir / Path(preferred_filename).name
    if direct_candidate.is_file() and direct_candidate.stat().st_size > 1024 * 1024:
        return str(direct_candidate)

    target_filename = preferred_filename
    if list_repo_files is not None:
        try:
            files = list_repo_files(repo_id)
            if preferred_filename not in files:
                import fnmatch
                matches = [f for f in files if fnmatch.fnmatch(f.lower(), fallback_glob.lower())]
                if matches:
                    target_filename = matches[0]
        except Exception:
            pass

    candidate_after_resolve = local_dir / Path(target_filename).name
    if candidate_after_resolve.is_file() and candidate_after_resolve.stat().st_size > 1024 * 1024:
        return str(candidate_after_resolve)

    print(f"[+] Memastikan file '{target_filename}' dari '{repo_id}'...", flush=True)
    if hf_hub_download is not None:
        try:
            dl = hf_hub_download(
                repo_id=repo_id,
                filename=target_filename,
                local_dir=str(local_dir),
            )
            return str(dl)
        except Exception as e:
            print(f"[!] Fallback wget untuk {target_filename}: {e}", flush=True)

    url = f"https://huggingface.co/{repo_id}/resolve/main/{target_filename}"
    out_path = local_dir / Path(target_filename).name
    subprocess.run(["wget", "-q", "-c", url, "-O", str(out_path)], check=True)
    return str(out_path)


def is_gguf_model_downloaded(model_key: str) -> Tuple[bool, float]:
    """Check whether both main GGUF and MMProj for a model exist on disk and return size in GB."""
    spec = get_model_spec(model_key)
    if spec.get("engine") == "pixai_tagger":
        return is_tagger_downloaded()

    main_dir = _repo_local_dir(spec["repo"])
    mmproj_dir = _repo_local_dir(spec["mmproj_repo"])

    main_file = main_dir / Path(spec["file"]).name
    mmproj_file = mmproj_dir / Path(spec["mmproj_file"]).name

    hf_cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    total_bytes = 0
    main_ok = False
    mmproj_ok = False

    if main_file.is_file() and main_file.stat().st_size > 1024 * 1024:
        main_ok = True
        total_bytes += main_file.stat().st_size
    elif hf_cache_root.exists():
        matches = list(hf_cache_root.rglob(Path(spec["file"]).name))
        if matches and matches[0].stat().st_size > 1024 * 1024:
            main_ok = True
            total_bytes += matches[0].stat().st_size

    if mmproj_file.is_file() and mmproj_file.stat().st_size > 1024 * 1024:
        mmproj_ok = True
        total_bytes += mmproj_file.stat().st_size
    elif hf_cache_root.exists():
        matches = list(hf_cache_root.rglob(Path(spec["mmproj_file"]).name))
        if matches and matches[0].stat().st_size > 1024 * 1024:
            mmproj_ok = True
            total_bytes += matches[0].stat().st_size

    return (main_ok and mmproj_ok), round(total_bytes / (1024 ** 3), 2)


def stop_llama_server() -> None:
    """Terminate active llama-server process and free GPU VRAM."""
    proc = ACTIVE_STATE.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    ACTIVE_STATE["proc"] = None
    ACTIVE_STATE["status"] = "STANDBY"
    ACTIVE_STATE["model_key"] = None

    if SERVER_PID_FILE.exists():
        try:
            pid = int(SERVER_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.8)
        except Exception:
            pass
        try:
            SERVER_PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass


def unload_vlm_server(include_tagger: bool = False) -> Dict[str, Any]:
    """Stop llama-server (and optionally PixAI Tagger) to release 100% of T4 VRAM."""
    with _SERVER_LOCK:
        stop_llama_server()
        if include_tagger:
            unload_pixai_tagger()
    return {"ok": True, "status": "STANDBY", "msg": "Model berhasil di-unload dari VRAM GPU."}


def is_server_healthy() -> bool:
    """Ping llama-server `/health` endpoint."""
    proc = ACTIVE_STATE.get("proc")
    if proc is None or proc.poll() is not None:
        return False
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_llama_server(model_key: str) -> Tuple[str, str]:
    """
    Thread-safe manager that downloads the requested GGUF + MMProj if needed and starts `llama-server`
    with Tesla T4 optimal parameters (`-c 8192 -np 1 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on`).
    """
    if model_key not in MODEL_REGISTRY:
        model_key = DEFAULT_MODEL_KEY

    spec = get_model_spec(model_key)
    if spec.get("engine") == "pixai_tagger":
        ensure_pixai_tagger()
        return str(TAGGER_STATE.get("weights_path") or ""), ""

    with _SERVER_LOCK:
        if ACTIVE_STATE.get("model_key") == model_key and is_server_healthy():
            ACTIVE_STATE["status"] = "ONLINE"
            return ACTIVE_STATE["model_path"], ACTIVE_STATE["mmproj_path"]

        ACTIVE_STATE["status"] = "LOADING"
        ACTIVE_STATE["last_error"] = None

        try:
            model_path = resolve_and_download_hf(spec["repo"], spec["file"], spec["file_pattern"])
            mmproj_path = resolve_and_download_hf(spec["mmproj_repo"], spec["mmproj_file"], spec["mmproj_pattern"])

            stop_llama_server()
            ACTIVE_STATE["status"] = "LOADING"

            server_bin, _, _ = get_llama_binaries()
            cmd = [
                server_bin,
                "-m", model_path,
                "--mmproj", mmproj_path,
                "-ngl", "99",
                "-c", "8192",
                "-np", "1",
                "-cb",
                "-ub", "512",
                "-b", "1024",
                "-t", "2",
                "--cache-type-k", "q8_0",
                "--cache-type-v", "q8_0",
                "--flash-attn", "on",
                "--host", "127.0.0.1",
                "--port", str(SERVER_PORT),
            ]
            for extra_flag in spec.get("server_flags", ["--jinja"]):
                if extra_flag not in cmd:
                    cmd.append(extra_flag)

            log_f = open(SERVER_LOG_FILE, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            SERVER_PID_FILE.write_text(str(proc.pid))

            for _ in range(150):
                if proc.poll() is not None:
                    break
                try:
                    r = requests.get(f"{SERVER_URL}/health", timeout=2)
                    if r.status_code == 200:
                        ACTIVE_STATE["model_key"] = model_key
                        ACTIVE_STATE["model_path"] = model_path
                        ACTIVE_STATE["mmproj_path"] = mmproj_path
                        ACTIVE_STATE["proc"] = proc
                        ACTIVE_STATE["status"] = "ONLINE"
                        print(f"[✓] llama-server siap dengan model: {model_key}", flush=True)
                        return model_path, mmproj_path
                except Exception:
                    pass
                time.sleep(1)

            tail_log = ""
            if SERVER_LOG_FILE.exists():
                tail_log = "\n".join(SERVER_LOG_FILE.read_text(errors="ignore").splitlines()[-25:])
            ACTIVE_STATE["status"] = "ERROR"
            ACTIVE_STATE["last_error"] = f"llama-server gagal mencapai status healthy.\n{tail_log}"
            ACTIVE_STATE["model_key"] = model_key
            ACTIVE_STATE["model_path"] = model_path
            ACTIVE_STATE["mmproj_path"] = mmproj_path
            return model_path, mmproj_path
        except Exception as e:
            ACTIVE_STATE["status"] = "ERROR"
            ACTIVE_STATE["last_error"] = str(e)
            raise


def image_to_data_uri(img_path: str, max_side: int = 1280) -> str:
    """Resize image in memory (without altering disk file) and encode as JPEG base64 Data URI."""
    with Image.open(img_path) as im:
        if im.mode in ("RGBA", "P"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            rgba = im.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[3])
            im = bg
        else:
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def build_routed_messages(
    model_key: str,
    sys_prompt: str,
    usr_prompt: str,
    image_data_uri: Optional[str] = None,
    subject_hint: str = "",
    hybrid_tag_context: str = "",
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible multimodal `messages` list tailored to each VLM's chat template:
    - `joycaption_mode`: System locked to `"You are a helpful image captioner."`, style instructions placed in user prompt.
    - `user_merged` (Gemma-3): Merges system + user instructions into a single `user` role message for 100% adherence.
    - `system_and_user` (Qwen3-VL / Qwen3.5): Keeps distinct `system` and `user` roles.
    """
    spec = get_model_spec(model_key)
    routing = spec.get("prompt_routing", "system_and_user")

    clean_sys = (sys_prompt or "").strip()
    clean_usr = (usr_prompt or "Describe this image in detail.").strip()

    extra_context_blocks = []
    if subject_hint and subject_hint.strip():
        extra_context_blocks.append(
            f"[Global Subject / Concept Context — Must be respected in the caption: {subject_hint.strip()}]"
        )
    if hybrid_tag_context and hybrid_tag_context.strip():
        extra_context_blocks.append(hybrid_tag_context.strip())

    extra_block_str = "\n".join(extra_context_blocks)

    if routing == "joycaption_mode":
        locked_sys = spec.get("joycaption_system_prompt", "You are a helpful image captioner.")
        combined_usr = clean_sys if clean_sys else clean_usr
        if clean_sys and clean_usr and clean_usr.lower() != "describe this image in detail.":
            combined_usr = f"{clean_sys}\n\n{clean_usr}"
        if extra_block_str:
            combined_usr = f"{combined_usr}\n\n{extra_block_str}"

        u_content: List[Dict[str, Any]] = []
        if image_data_uri:
            u_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
        u_content.append({"type": "text", "text": combined_usr})
        return [
            {"role": "system", "content": locked_sys},
            {"role": "user", "content": u_content},
        ]

    if routing == "user_merged":
        merged_text = f"{clean_sys}\n\n{clean_usr}".strip() if clean_sys else clean_usr
        if extra_block_str:
            merged_text = f"{merged_text}\n\n{extra_block_str}"
        u_content = []
        if image_data_uri:
            u_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
        u_content.append({"type": "text", "text": merged_text})
        return [{"role": "user", "content": u_content}]

    # Default: system_and_user (Qwen3-VL / Qwen3.5)
    final_usr = f"{clean_usr}\n\n{extra_block_str}".strip() if extra_block_str else clean_usr
    u_content = []
    if image_data_uri:
        u_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
    u_content.append({"type": "text", "text": final_usr})

    messages: List[Dict[str, Any]] = []
    if clean_sys:
        messages.append({"role": "system", "content": clean_sys})
    messages.append({"role": "user", "content": u_content})
    return messages


def _run_mtmd_cli_fallback(
    img_path: str,
    model_path: str,
    mmproj_path: str,
    sys_prompt: str,
    usr_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Fallback execution using `llama-mtmd-cli` with combined `-p` prompt (compatible with all libmtmd releases).
    """
    _, mtmd_bin, cli_bin = get_llama_binaries()
    runner_bin = mtmd_bin or cli_bin
    combined_prompt = f"{sys_prompt.strip()}\n\n{usr_prompt.strip()}".strip()

    cmd = [
        runner_bin,
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "-ngl", "99",
        "-c", "4096",
        "-t", "2",
        "--flash-attn", "on",
        "--image", str(img_path),
        "-p", combined_prompt,
        "-n", str(int(max_tokens)),
        "--temp", str(float(temperature)),
    ]
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    return clean_vlm_response(res.stdout, sys_prompt, usr_prompt)


def run_vlm_caption_single(
    img_path: str,
    model_key: str,
    sys_prompt: str,
    usr_prompt: str = "Describe this image in detail.",
    max_tokens: int = 220,
    temperature: Optional[float] = None,
    subject_hint: str = "",
    hybrid_tagger_enabled: bool = False,
    hybrid_mode: str = "grounding",  # grounding | prepend | append
    general_threshold: float = 0.30,
    character_threshold: float = 0.75,
    replace_underscore: bool = False,
    escape_parens: bool = False,
    exclude_tags: str = "",
) -> Dict[str, Any]:
    """
    Execute single-image captioning or tagging:
    - If `model_key == PIXAI_TAGGER_KEY`, runs standalone ultra-fast PixAI Tagger v0.9.
    - Otherwise runs VLM via `llama-server` (with optional Hybrid PixAI Tagger v0.9 assistance).
    """
    spec = get_model_spec(model_key)

    # 1. Standalone PixAI Tagger v0.9 Mode
    if spec.get("engine") == "pixai_tagger":
        tag_res = run_pixai_tagger(
            img_path=img_path,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            include_ip=True,
            replace_underscore=replace_underscore,
            escape_parens=escape_parens,
            exclude_tags=exclude_tags,
        )
        cap = tag_res["caption"]
        stats = estimate_token_count(cap)
        return {
            "ok": bool(cap),
            "caption": cap,
            "words": stats["words"],
            "tokens": stats["tokens"],
            "tagger_meta": tag_res,
        }

    # 2. Optional Hybrid PixAI Tagger Pre-Pass
    tag_res: Optional[Dict[str, Any]] = None
    hybrid_context_str = ""
    if hybrid_tagger_enabled:
        try:
            tag_res = run_pixai_tagger(
                img_path=img_path,
                general_threshold=general_threshold,
                character_threshold=character_threshold,
                include_ip=True,
                replace_underscore=replace_underscore,
                escape_parens=escape_parens,
                exclude_tags=exclude_tags,
            )
            if hybrid_mode in ("grounding", "both"):
                hybrid_context_str = build_hybrid_vlm_context(tag_res)
        except Exception as e:
            print(f"[!] Peringatan Hybrid PixAI Tagger ({e}), melanjutkan dengan VLM murni...", flush=True)

    # 3. Ensure llama-server is running with requested VLM
    model_path, mmproj_path = ensure_llama_server(model_key)
    sampling = dict(spec.get("caption_sampling", {}))
    eff_temp = float(temperature) if temperature is not None else float(sampling.get("temperature", 0.3))

    data_uri = image_to_data_uri(img_path)
    messages = build_routed_messages(
        model_key=model_key,
        sys_prompt=sys_prompt,
        usr_prompt=usr_prompt,
        image_data_uri=data_uri,
        subject_hint=subject_hint,
        hybrid_tag_context=hybrid_context_str,
    )

    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": eff_temp,
        "top_p": float(sampling.get("top_p", 0.9)),
        "top_k": int(sampling.get("top_k", 40)),
        "repeat_penalty": float(sampling.get("repeat_penalty", 1.05)),
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }
    if "min_p" in sampling:
        payload["min_p"] = float(sampling["min_p"])
    if "presence_penalty" in sampling:
        payload["presence_penalty"] = float(sampling["presence_penalty"])
    if "reasoning_budget" in sampling:
        payload["reasoning_budget"] = int(sampling["reasoning_budget"])

    cleaned_caption = ""
    with _INFERENCE_LOCK:
        for attempt in range(2):
            try:
                r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=180)
                if r.ok:
                    resp_json = r.json()
                    msg_obj = resp_json["choices"][0]["message"]
                    raw_text = extract_message_text(msg_obj, allow_reasoning_fallback=True)
                    cleaned_caption = clean_vlm_response(raw_text, sys_prompt, usr_prompt)
                    if cleaned_caption:
                        break
            except Exception as e:
                print(f"[!] Request llama-server percobaan {attempt + 1} gagal ({e})", flush=True)
                if attempt == 0:
                    stop_llama_server()
                    ensure_llama_server(model_key)

        if not cleaned_caption:
            print("[!] Menjalankan fallback llama-mtmd-cli...", flush=True)
            stop_llama_server()
            cleaned_caption = _run_mtmd_cli_fallback(
                img_path, model_path, mmproj_path, sys_prompt, usr_prompt, int(max_tokens), eff_temp
            )

    # 4. Merge Hybrid Tags if prepend/append/both was selected
    if tag_res and tag_res.get("caption") and cleaned_caption:
        tag_str = tag_res["caption"].strip()
        if hybrid_mode in ("prepend", "both"):
            cleaned_caption = f"{tag_str}. {cleaned_caption}"
        elif hybrid_mode == "append":
            cleaned_caption = f"{cleaned_caption.rstrip('. ')}. Tags: {tag_str}"

    stats = estimate_token_count(cleaned_caption)
    return {
        "ok": bool(cleaned_caption),
        "caption": cleaned_caption,
        "words": stats["words"],
        "tokens": stats["tokens"],
        "tagger_meta": tag_res,
    }


def run_multimodal_chat(
    model_key: str,
    sys_prompt: str,
    history: List[Dict[str, Any]],
    user_message: str,
    image_data_uri: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 768,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """
    Execute multi-turn multimodal chat.
    Preserves up to the 2 most recent images in conversation history so follow-up questions
    continue to see the image without overflowing the 8192 context window on Tesla T4.
    """
    spec = get_model_spec(model_key)
    if spec.get("engine") == "pixai_tagger":
        model_key = DEFAULT_MODEL_KEY
        spec = get_model_spec(model_key)

    ensure_llama_server(model_key)
    routing = spec.get("prompt_routing", "system_and_user")

    turns_with_img = [idx for idx, h in enumerate(history) if h.get("image_data_uri")]
    keep_img_indices = set(turns_with_img[-1:] if image_data_uri else turns_with_img[-2:])

    messages: List[Dict[str, Any]] = []
    if routing != "user_merged" and sys_prompt.strip():
        messages.append({"role": "system", "content": sys_prompt.strip()})

    first_user_merged = False
    for idx, h in enumerate(history):
        role = h.get("role", "user")
        text_val = (h.get("content") or "").strip()
        h_img = h.get("image_data_uri") if idx in keep_img_indices else None

        if role == "user":
            if routing == "user_merged" and not first_user_merged and sys_prompt.strip():
                text_val = f"{sys_prompt.strip()}\n\n{text_val}".strip()
                first_user_merged = True
            if h_img:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": h_img}},
                        {"type": "text", "text": text_val or "Describe this image."},
                    ],
                })
            else:
                messages.append({"role": "user", "content": text_val})
        else:
            messages.append({"role": "assistant", "content": text_val})

    cur_text = (user_message or "Describe this image in detail.").strip()
    if routing == "user_merged" and not first_user_merged and sys_prompt.strip():
        cur_text = f"{sys_prompt.strip()}\n\n{cur_text}".strip()

    u_content: List[Dict[str, Any]] = []
    if image_data_uri:
        u_content.append({"type": "image_url", "image_url": {"url": image_data_uri}})
    u_content.append({"type": "text", "text": cur_text})
    messages.append({"role": "user", "content": u_content})

    use_thinking = bool(enable_thinking and spec.get("family") == "qwen35")
    payload: Dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "chat_template_kwargs": {"enable_thinking": use_thinking},
        "reasoning_format": "deepseek" if use_thinking else "none",
    }
    if not use_thinking:
        payload["reasoning_budget"] = 0

    with _INFERENCE_LOCK:
        try:
            r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=180)
            r.raise_for_status()
            msg_obj = r.json()["choices"][0]["message"]
            raw_reply = extract_message_text(msg_obj, allow_reasoning_fallback=True)
            thinking_trace = (msg_obj.get("reasoning_content") or "").strip() if use_thinking else ""
            cleaned_reply = clean_vlm_response(
                raw_reply, strip_meta_preamble=False, single_line=False
            )
            return {
                "ok": True,
                "reply": cleaned_reply,
                "thinking": thinking_trace,
                "model_used": model_key,
            }
        except Exception as e:
            return {
                "ok": False,
                "reply": f"⚠️ Error saat menghubungi mesin VLM: {e}",
                "thinking": "",
                "model_used": model_key,
            }


def list_downloaded_models() -> List[Dict[str, Any]]:
    """Return status and disk usage of all 8 models in MODEL_REGISTRY."""
    results = []
    for key, spec in MODEL_REGISTRY.items():
        downloaded, size_gb = is_gguf_model_downloaded(key)
        is_active = (
            (ACTIVE_STATE.get("model_key") == key and is_server_healthy())
            if spec.get("engine") == "llama_cpp"
            else (TAGGER_STATE.get("status") == "ONLINE")
        )
        results.append({
            "key": key,
            "engine": spec.get("engine", "llama_cpp"),
            "family": spec.get("family", ""),
            "desc": spec.get("desc", ""),
            "downloaded": downloaded,
            "size_gb": size_gb if downloaded else spec.get("approx_size_gb", 5.8),
            "active_in_vram": is_active,
            "sampling": spec.get("caption_sampling", {}),
        })
    return results


def delete_downloaded_model(model_key: str) -> Dict[str, Any]:
    """Delete cached model files on disk for `model_key` (unloading from VRAM first if active)."""
    if model_key not in MODEL_REGISTRY:
        return {"ok": False, "msg": f"Model '{model_key}' tidak dikenal."}

    spec = MODEL_REGISTRY[model_key]
    if spec.get("engine") == "pixai_tagger":
        unload_pixai_tagger()
        freed_bytes = 0
        if TAGGER_CACHE_DIR.exists():
            for p in TAGGER_CACHE_DIR.rglob("*"):
                if p.is_file():
                    freed_bytes += p.stat().st_size
            shutil.rmtree(TAGGER_CACHE_DIR, ignore_errors=True)
        return {
            "ok": True,
            "msg": f"Berhasil menghapus cache PixAI Tagger v0.9 ({freed_bytes / (1024**3):.2f} GB dibebaskan).",
        }

    if ACTIVE_STATE.get("model_key") == model_key:
        stop_llama_server()

    freed_bytes = 0
    target_names = {Path(spec["file"]).name, Path(spec["mmproj_file"]).name}

    repo_dir = _repo_local_dir(spec["repo"])
    if repo_dir.exists():
        for p in repo_dir.rglob("*"):
            if p.is_file() and (p.name == Path(spec["file"]).name or spec["repo"] != spec["mmproj_repo"]):
                freed_bytes += p.stat().st_size
                p.unlink(missing_ok=True)

    hf_cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache_root.exists():
        for fname in target_names:
            for m in hf_cache_root.rglob(fname):
                try:
                    real_p = m.resolve()
                    if real_p.is_file():
                        freed_bytes += real_p.stat().st_size
                        real_p.unlink(missing_ok=True)
                    m.unlink(missing_ok=True)
                except Exception:
                    pass

    return {
        "ok": True,
        "msg": f"Berhasil menghapus file GGUF '{model_key.split('[')[0].strip()}' ({freed_bytes / (1024**3):.2f} GB dibebaskan).",
    }


def get_system_telemetry() -> Dict[str, Any]:
    """
    Collect live GPU VRAM, System RAM, Disk Free Space, llama-server status,
    PixAI Tagger status, and per-model disk cache inventory.
    """
    vram_used_gb = 0.0
    vram_total_gb = 15.0
    gpu_name = "NVIDIA Tesla T4"

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [x.strip() for x in res.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 3:
                gpu_name = parts[0]
                vram_used_gb = round(float(parts[1]) / 1024.0, 2)
                vram_total_gb = round(float(parts[2]) / 1024.0, 2)
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                free_b, tot_b = torch.cuda.mem_get_info(0)
                vram_total_gb = round(tot_b / (1024 ** 3), 2)
                vram_used_gb = round((tot_b - free_b) / (1024 ** 3), 2)
        except Exception:
            pass

    check_disk_path = "/content" if os.path.exists("/content") else "/"
    usage = shutil.disk_usage(check_disk_path)
    disk_free_gb = round(usage.free / (1024 ** 3), 2)
    disk_total_gb = round(usage.total / (1024 ** 3), 2)

    srv_status = ACTIVE_STATE.get("status", "STANDBY")
    if srv_status == "ONLINE" and not is_server_healthy():
        srv_status = "STANDBY"
        ACTIVE_STATE["status"] = "STANDBY"

    return {
        "server_status": srv_status,
        "active_model": ACTIVE_STATE.get("model_key"),
        "tagger_status": TAGGER_STATE.get("status", "STANDBY"),
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "disk_free_gb": disk_free_gb,
        "disk_total_gb": disk_total_gb,
        "models_inventory": list_downloaded_models(),
    }
