"""
core/tagger_manager.py
Native PyTorch + timm implementation for `pixai-labs/pixai-tagger-v0.9`.
Supports:
1. Standalone Ultra-Fast Booru Tagging (13,461 tags: 9,741 General + 3,720 Character + Auto IP/Copyright Mapping).
2. Hybrid Tag-Assisted VLM Captioning (Grounding VLM prompts with character/IP/visual tags or merging tags with NL captions).
3. Automatic zero-token fallback from gated `pixai-labs/pixai-tagger-v0.9` to ungated mirror `yabaiyobai/pixai-tagger-v0.9-private`.
"""

import os
import re
import gc
import json
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from config.vlm_catalog import MODEL_REGISTRY, PIXAI_TAGGER_KEY

TAGGER_CACHE_DIR = Path(os.environ.get("VLM_MODELS_DIR", "/content/vlm_models")) / "pixai_tagger_v09"
_TAGGER_LOCK = threading.Lock()

TAGGER_STATE: Dict[str, Any] = {
    "status": "STANDBY",  # STANDBY | LOADING | ONLINE | ERROR
    "device": None,
    "model": None,
    "transform": None,
    "index_to_tag": {},
    "gen_tag_count": 9741,
    "char_tag_count": 3720,
    "char_ip_map": {},
    "weights_path": None,
    "error": None,
}


def _pil_to_rgb_white_bg(image: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Apply EXIF orientation transpose and composite RGBA/P transparency onto a clean white RGB background."""
    image.load()
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, bg_color)
        bg.paste(image, mask=image.split()[3])
        return bg
    if image.mode in ("P", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, bg_color)
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return image.convert("RGB")


def _download_tagger_file(filename: str) -> Path:
    """
    Download required PixAI Tagger v0.9 file (`model_v0.9.pth`, `tags_v0.9_13k.json`, `char_ip_map.json`).
    Tries official `pixai-labs/pixai-tagger-v0.9` first, then automatically falls back to
    the ungated mirror `yabaiyobai/pixai-tagger-v0.9-private` if no HF_TOKEN or gated access is present.
    """
    TAGGER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_target = TAGGER_CACHE_DIR / filename
    if local_target.is_file() and local_target.stat().st_size > 1024:
        return local_target

    spec = MODEL_REGISTRY[PIXAI_TAGGER_KEY]
    repos_to_try = [spec["repo"], spec["fallback_repo"]]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_hub_download is not None:
        for repo_id in repos_to_try:
            try:
                print(f"[+] Mengunduh komponen PixAI Tagger '{filename}' dari '{repo_id}'...", flush=True)
                dl_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(TAGGER_CACHE_DIR),
                    token=hf_token if repo_id == spec["repo"] else None,
                )
                return Path(dl_path)
            except Exception as e:
                print(f"[!] Info unduhan '{filename}' dari '{repo_id}' ({e}), mencoba mirror cadangan...", flush=True)

    fallback_url = f"https://huggingface.co/{spec['fallback_repo']}/resolve/main/{filename}"
    print(f"[+] Mengunduh via wget fallback: {fallback_url}", flush=True)
    subprocess.run(["wget", "-q", "-c", fallback_url, "-O", str(local_target)], check=True)
    return local_target


def is_tagger_downloaded() -> Tuple[bool, float]:
    """Return whether PixAI Tagger v0.9 files exist on disk and their total size in GB."""
    if not TAGGER_CACHE_DIR.exists():
        return False, 0.0
    req = ["model_v0.9.pth", "tags_v0.9_13k.json", "char_ip_map.json"]
    total_bytes = 0
    all_exist = True
    for fname in req:
        p = TAGGER_CACHE_DIR / fname
        if p.is_file() and p.stat().st_size > 100:
            total_bytes += p.stat().st_size
        else:
            all_exist = False
    return all_exist, round(total_bytes / (1024 ** 3), 2)


def ensure_pixai_tagger() -> Dict[str, Any]:
    """
    Build and load PixAI Tagger v0.9 (`EVA02-Large` + `TaggingHead(1024, 13461)`) into GPU/CPU memory.
    Thread-safe and cached across calls.
    """
    with _TAGGER_LOCK:
        if TAGGER_STATE["status"] == "ONLINE" and TAGGER_STATE["model"] is not None:
            return TAGGER_STATE

        TAGGER_STATE["status"] = "LOADING"
        TAGGER_STATE["error"] = None
        try:
            import torch
            import timm
            import torchvision.transforms as transforms

            weights_file = _download_tagger_file("model_v0.9.pth")
            tags_file = _download_tagger_file("tags_v0.9_13k.json")
            mapping_file = _download_tagger_file("char_ip_map.json")

            with tags_file.open("r", encoding="utf-8") as f:
                tag_info = json.load(f)
            tag_map: Dict[str, int] = tag_info["tag_map"]
            tag_split: Dict[str, int] = tag_info["tag_split"]
            gen_tag_count = int(tag_split.get("gen_tag_count", 9741))
            char_tag_count = int(tag_split.get("character_tag_count", 3720))
            total_classes = gen_tag_count + char_tag_count

            with mapping_file.open("r", encoding="utf-8") as f:
                char_ip_map = json.load(f)

            class TaggingHead(torch.nn.Module):
                def __init__(self, input_dim: int, num_classes: int):
                    super().__init__()
                    self.head = torch.nn.Sequential(torch.nn.Linear(input_dim, num_classes))

                def forward(self, x):
                    logits = self.head(x)
                    return torch.sigmoid(logits)

            spec = MODEL_REGISTRY[PIXAI_TAGGER_KEY]
            encoder = timm.create_model(spec["encoder_repo"], pretrained=False)
            encoder.reset_classifier(0)
            decoder = TaggingHead(1024, total_classes)
            model = torch.nn.Sequential(encoder, decoder)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                state = torch.load(str(weights_file), map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(str(weights_file), map_location=device)
            model.load_state_dict(state)
            model.to(device).eval()

            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            TAGGER_STATE["model"] = model
            TAGGER_STATE["device"] = device
            TAGGER_STATE["transform"] = transform
            TAGGER_STATE["index_to_tag"] = {int(v): k for k, v in tag_map.items()}
            TAGGER_STATE["gen_tag_count"] = gen_tag_count
            TAGGER_STATE["char_tag_count"] = char_tag_count
            TAGGER_STATE["char_ip_map"] = char_ip_map
            TAGGER_STATE["weights_path"] = str(weights_file)
            TAGGER_STATE["status"] = "ONLINE"
            print(f"[✓] PixAI Tagger v0.9 aktif di {device.upper()} ({total_classes} tags).", flush=True)
            return TAGGER_STATE
        except Exception as e:
            TAGGER_STATE["status"] = "ERROR"
            TAGGER_STATE["error"] = str(e)
            print(f"[!] Gagal memuat PixAI Tagger v0.9: {e}", flush=True)
            raise


def unload_pixai_tagger() -> bool:
    """Unload PixAI Tagger v0.9 from GPU VRAM and run garbage collection."""
    with _TAGGER_LOCK:
        TAGGER_STATE["model"] = None
        TAGGER_STATE["transform"] = None
        TAGGER_STATE["status"] = "STANDBY"
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print("[✓] PixAI Tagger v0.9 berhasil di-unload dari VRAM.", flush=True)
        return True


def format_tag_token(tag: str, replace_underscore: bool = False, escape_parens: bool = False) -> str:
    """Format a single Danbooru tag according to user preferences."""
    t = tag.strip()
    if replace_underscore and not re.match(r"^[:;><\^v0-9_]{2,4}$", t):
        t = t.replace("_", " ")
    if escape_parens:
        t = t.replace("(", r"\(").replace(")", r"\)")
    return t


def run_pixai_tagger(
    img_path: str,
    general_threshold: float = 0.30,
    character_threshold: float = 0.75,
    top_k: int = 128,
    include_ip: bool = True,
    replace_underscore: bool = False,
    escape_parens: bool = False,
    exclude_tags: str = "",
) -> Dict[str, Any]:
    """
    Run PixAI Tagger v0.9 inference on a single image path.
    Returns structured dict with `caption`, `general_tags`, `character_tags`, `ip_tags`, and `scores`.
    """
    import torch

    state = ensure_pixai_tagger()
    model = state["model"]
    device = state["device"]
    transform = state["transform"]
    idx_to_tag = state["index_to_tag"]
    gen_count = state["gen_tag_count"]
    char_ip_map = state["char_ip_map"]

    excluded_set = set()
    if exclude_tags:
        for raw_ex in exclude_tags.split(","):
            norm_ex = raw_ex.strip().lower()
            if norm_ex:
                excluded_set.add(norm_ex)
                excluded_set.add(norm_ex.replace(" ", "_"))
                excluded_set.add(norm_ex.replace("_", " "))

    with Image.open(img_path) as raw_im:
        rgb_im = _pil_to_rgb_white_bg(raw_im)
        tensor = transform(rgb_im).unsqueeze(0)

    with torch.inference_mode():
        if device == "cuda":
            tensor = tensor.pin_memory().to(device, non_blocking=True)
        else:
            tensor = tensor.to(device)
        probs = model(tensor)[0].detach().cpu()

    gen_probs = probs[:gen_count]
    char_probs = probs[gen_count:]

    gen_mask = gen_probs >= float(general_threshold)
    char_mask = char_probs >= float(character_threshold)

    gen_indices = gen_mask.nonzero(as_tuple=True)[0].tolist()
    char_indices = (char_mask.nonzero(as_tuple=True)[0] + gen_count).tolist()

    gen_Pairs = sorted(
        [(idx_to_tag[i], float(probs[i].item())) for i in gen_indices if i in idx_to_tag],
        key=lambda x: x[1],
        reverse=True,
    )
    if top_k and int(top_k) > 0:
        gen_Pairs = gen_Pairs[: int(top_k)]

    char_Pairs = sorted(
        [(idx_to_tag[i], float(probs[i].item())) for i in char_indices if i in idx_to_tag],
        key=lambda x: x[1],
        reverse=True,
    )

    raw_char_tags = [t for t, _ in char_Pairs if t.lower() not in excluded_set]
    raw_gen_tags = [t for t, _ in gen_Pairs if t.lower() not in excluded_set]

    raw_ip_tags: List[str] = []
    if include_ip:
        seen_ips = set()
        for c_tag in raw_char_tags:
            for ip in char_ip_map.get(c_tag, []):
                if ip.lower() not in excluded_set and ip not in seen_ips:
                    seen_ips.add(ip)
                    raw_ip_tags.append(ip)

    formatted_char = [format_tag_token(t, replace_underscore, escape_parens) for t in raw_char_tags]
    formatted_ip = [format_tag_token(t, replace_underscore, escape_parens) for t in raw_ip_tags]
    formatted_gen = [format_tag_token(t, replace_underscore, escape_parens) for t in raw_gen_tags]

    combined: List[str] = []
    seen_out = set()
    for t in formatted_char + formatted_ip + formatted_gen:
        low_t = t.lower()
        if low_t not in seen_out:
            seen_out.add(low_t)
            combined.append(t)

    caption_str = ", ".join(combined)
    return {
        "caption": caption_str,
        "character_tags": formatted_char,
        "ip_tags": formatted_ip,
        "general_tags": formatted_gen,
        "scores": {
            "character": {t: round(s, 4) for t, s in char_Pairs[:15]},
            "general": {t: round(s, 4) for t, s in gen_Pairs[:30]},
        },
    }


def build_hybrid_vlm_context(tag_result: Dict[str, Any]) -> str:
    """
    Build a concise grounding hint from PixAI Tagger v0.9 results to inject into the VLM prompt
    so the VLM accurately names the anime character, copyright series, and visual traits.
    """
    chars = tag_result.get("character_tags", [])
    ips = tag_result.get("ip_tags", [])
    gens = tag_result.get("general_tags", [])[:28]

    parts = []
    if chars:
        parts.append(f"Detected Character(s): {', '.join(chars)}")
    if ips:
        parts.append(f"Source Series/Copyright: {', '.join(ips)}")
    if gens:
        parts.append(f"Key Visual Attributes: {', '.join(gens)}")

    if not parts:
        return ""
    return (
        "[Reference Visual Tags from PixAI Tagger v0.9 — Incorporate accurate character names, series, "
        "and key traits naturally into your response without copying as a raw list unless requested]: "
        + " | ".join(parts)
    )
