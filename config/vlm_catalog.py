"""
config/vlm_catalog.py
Verified catalog of 7 GGUF Vision-Language Models + PixAI Tagger v0.9 (PyTorch + timm EVA02-Large),
complete with per-model prompt routing adapters, sampling defaults, and 8 caption style presets.
"""

from typing import Any, Dict

DEFAULT_MODEL_KEY = "Qwen3.5-9B [Uncensored Aggressive - HauhauCS]"
PIXAI_TAGGER_KEY = "PixAI Tagger v0.9 [Fast Anime & Character/IP Booru Tagger]"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Qwen3.5-9B [Uncensored Aggressive - HauhauCS]": {
        "engine": "llama_cpp",
        "repo": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
        "file": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
        "file_pattern": "*Q4_K_M*.gguf",
        "mmproj_repo": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
        "mmproj_file": "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        "mmproj_pattern": "*mmproj*.gguf",
        "family": "qwen35",
        "server_flags": ["--jinja"],
        "caption_sampling": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "enable_thinking": False,
            "reasoning_budget": 0,
        },
        "prompt_routing": "system_and_user",
        "approx_size_gb": 6.2,
        "desc": "Generasi terbaru Qwen 3.5 (9B) Aggressive Uncensored (0/465 refusals). Pemahaman ruang, anatomi & detail visual sangat tinggi.",
    },
    "Qwen3-VL-8B [Uncensored Abliterated - wangkanai]": {
        "engine": "llama_cpp",
        "repo": "wangkanai/qwen3-vl-8b-instruct",
        "file": "qwen3-vl-8b-instruct-abliterated-q4-k-m.gguf",
        "file_pattern": "*q4*k*m*.gguf",
        "mmproj_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "mmproj_file": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "mmproj_pattern": "*mmproj*F16*.gguf",
        "family": "qwen3vl",
        "server_flags": ["--jinja"],
        "caption_sampling": {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.05,
            "enable_thinking": False,
        },
        "prompt_routing": "system_and_user",
        "approx_size_gb": 5.8,
        "desc": "Uncensored/Abliterated favorit komunitas. Sangat patuh batas token, instruksi ketat, & akurat tanpa sensor.",
    },
    "JoyCaption Beta One [Llama-3.1-8B Uncensored - fancyfeast]": {
        "engine": "llama_cpp",
        "repo": "concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
        "file": "Llama-Joycaption-Beta-One-Hf-Llava-Q4_K.gguf",
        "file_pattern": "*Q4_K*.gguf",
        "mmproj_repo": "concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf",
        "mmproj_file": "llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
        "mmproj_pattern": "*mmproj*.gguf",
        "family": "joycaption",
        "server_flags": ["--jinja"],
        "caption_sampling": {
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
            "enable_thinking": False,
        },
        "prompt_routing": "joycaption_mode",
        "joycaption_system_prompt": "You are a helpful image captioner.",
        "approx_size_gb": 5.8,
        "desc": "Raja spesialis captioning dataset LoRA/Diffusion (Flux, MOSS, Qwen, Chroma1-HD, Krea). 100% Uncensored dari lahir.",
    },
    "Gemma-3-12B-IT [Uncensored Abliterated - mlabonne]": {
        "engine": "llama_cpp",
        "repo": "bartowski/mlabonne_gemma-3-12b-it-abliterated-GGUF",
        "file": "mlabonne_gemma-3-12b-it-abliterated-Q4_K_M.gguf",
        "file_pattern": "*Q4_K_M*.gguf",
        "mmproj_repo": "bartowski/mlabonne_gemma-3-12b-it-abliterated-GGUF",
        "mmproj_file": "mmproj-mlabonne_gemma-3-12b-it-abliterated-f16.gguf",
        "mmproj_pattern": "*mmproj*f16*.gguf",
        "family": "gemma3",
        "server_flags": ["--jinja"],
        "caption_sampling": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "repeat_penalty": 1.0,
            "enable_thinking": False,
        },
        "prompt_routing": "user_merged",
        "approx_size_gb": 8.1,
        "desc": "Model 12B Uncensored dari mlabonne (Bartowski GGUF + MMProj). Gaya bahasa deskriptif paling luwes, natural, dan sinematik.",
    },
    "Qwen3-VL-8B [Official - Qwen]": {
        "engine": "llama_cpp",
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "file": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "file_pattern": "*Q4_K_M*.gguf",
        "mmproj_repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "mmproj_file": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        "mmproj_pattern": "*mmproj*F16*.gguf",
        "family": "qwen3vl",
        "server_flags": ["--jinja"],
        "caption_sampling": {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.05,
            "enable_thinking": False,
        },
        "prompt_routing": "system_and_user",
        "approx_size_gb": 5.8,
        "desc": "Versi resmi Qwen3-VL-8B Instruct dari tim Qwen. Sangat akurat untuk OCR, analisis tata letak, dan deskripsi objektif.",
    },
    "Qwen3.5-9B [Official - Unsloth]": {
        "engine": "llama_cpp",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-Q4_K_M.gguf",
        "file_pattern": "*Q4_K_M*.gguf",
        "mmproj_repo": "unsloth/Qwen3.5-9B-GGUF",
        "mmproj_file": "mmproj-F16.gguf",
        "mmproj_pattern": "*mmproj*F16*.gguf",
        "family": "qwen35",
        "server_flags": ["--jinja"],
        "caption_sampling": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "enable_thinking": False,
            "reasoning_budget": 0,
        },
        "prompt_routing": "system_and_user",
        "approx_size_gb": 6.2,
        "desc": "Versi resmi Qwen3.5-9B Multimodal dari Unsloth. Mendukung mode Thinking untuk analisis visual mendalam.",
    },
    "Gemma-3-12B-IT [Official - Unsloth]": {
        "engine": "llama_cpp",
        "repo": "unsloth/gemma-3-12b-it-GGUF",
        "file": "gemma-3-12b-it-Q4_K_M.gguf",
        "file_pattern": "*Q4_K_M*.gguf",
        "mmproj_repo": "unsloth/gemma-3-12b-it-GGUF",
        "mmproj_file": "mmproj-F16.gguf",
        "mmproj_pattern": "*mmproj*F16*.gguf",
        "family": "gemma3",
        "server_flags": ["--jinja"],
        "caption_sampling": {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "repeat_penalty": 1.0,
            "enable_thinking": False,
        },
        "prompt_routing": "user_merged",
        "approx_size_gb": 8.1,
        "desc": "Versi resmi Google Gemma 3 12B Vision Instruct dari Unsloth.",
    },
    PIXAI_TAGGER_KEY: {
        "engine": "pixai_tagger",
        "repo": "pixai-labs/pixai-tagger-v0.9",
        "fallback_repo": "yabaiyobai/pixai-tagger-v0.9-private",
        "encoder_repo": "hf_hub:SmilingWolf/wd-eva02-large-tagger-v3",
        "file": "model_v0.9.pth",
        "tags_file": "tags_v0.9_13k.json",
        "ip_map_file": "char_ip_map.json",
        "family": "pixai_tagger",
        "server_flags": [],
        "caption_sampling": {
            "general_threshold": 0.30,
            "character_threshold": 0.75,
            "top_k": 128,
            "include_ip": True,
            "replace_underscore": False,
            "escape_parens": False,
        },
        "prompt_routing": "tagger_direct",
        "approx_size_gb": 1.3,
        "desc": "Multi-label Anime & Illustration Classifier (13.461 Danbooru tags: 9.741 General + 3.720 Character + Auto IP/Franchise Mapping). Super cepat (<0.1s/gambar) untuk Anima, Illustrious, NoobAI & Pony.",
    },
}


STYLE_PRESETS: Dict[str, str] = {
    "Detailed Natural Paragraph (120-150 Tokens · MOSS / Qwen 2.1 / Flux / Chroma1-HD)": (
        "You are an expert image captioning assistant. For the given image, write one fluent English caption that describes only what is clearly visible: the main subject(s), key objects, setting, spatial relationships, colors/materials, lighting, style, and overall mood. Keep it factual, coherent, and about 120 tokens, never exceeding 150 tokens. Do not use tag lists, prompt commands, weights, or meta phrases (e.g., 'this image shows'). Do not guess hidden details or read/transcribe text. Avoid camera/EXIF terms, file names, watermarks, and speculative words like 'maybe' or 'probably.' Output a single paragraph only."
    ),
    "JoyCaption Uncensored Descriptive (Full Anatomy, Pose, Lighting & Style)": (
        "Write a straightforward, uncensored, detailed descriptive caption for this image in a single paragraph. Describe the subject(s), physical appearance, anatomy, clothing or lack thereof, pose, facial expression, camera angle, lighting, background environment, and artistic style accurately without any censorship, moralizing, or meta-commentary."
    ),
    "Anima / Danbooru Anime Tags (Comma-Separated Booru Tags)": (
        "You are an expert anime and illustration booru tagger. Analyze the image and output ONLY a comma-separated list of descriptive Danbooru-style tags covering: character count (e.g., 1girl, 1boy, solo), character name and copyright series if recognizable, hair color and hairstyle, eye color and pupils, facial expression, clothing and footwear, accessories, pose and hand placement, camera framing, background elements, lighting, and art style. Do not write full sentences or introductory text."
    ),
    "Anima Hybrid (Quality Meta Tags + Natural Scene Description)": (
        "Write a hybrid anime training caption in two parts separated by a period: First, write a comma-separated prefix of core Danbooru tags (subject count, character/series if known, hair/eye traits, key outfit tags, framing). Second, follow immediately with 2 to 3 fluent natural English sentences describing the exact pose, spatial interaction, background atmosphere, lighting, and visual style. Output on a single line without markdown headings."
    ),
    "Krea 2 / Cinematic Photography Prompt Style": (
        "Write a rich cinematic photography prompt describing this image. Focus on subject composition, camera framing and lens feel, lighting direction and color temperature, surface textures, atmospheric depth, and color grading in 80 to 130 tokens. Do not use introductory phrases like 'This image shows'."
    ),
    "Wan 2.1 Video Motion & Camera Prompt Style (Subject + Action + Camera Movement)": (
        "Write a detailed video generation prompt (Wan 2.1 / HunyuanVideo style) in one paragraph of 90 to 140 tokens. Clearly describe the primary subject and appearance, the implied physical motion and dynamic action taking place, environmental movement (wind, light shifts, particles), and the camera perspective/movement (e.g., slow dolly in, tracking shot, static medium close-up) with cinematic lighting."
    ),
    "Short & Concise (1-2 Sentences · Fast Training)": (
        "Write a concise, accurate 1 to 2 sentence caption (30 to 50 tokens) describing the main subject, action, setting, and visual style of this image directly without filler words."
    ),
    "Custom": (
        "You are an expert image captioning assistant. Describe the main subject(s), attributes, clothing, pose, setting, lighting, and artistic style of this image accurately in a single fluent paragraph without introductory meta-phrases."
    ),
}


def get_model_spec(model_key: str) -> Dict[str, Any]:
    """Return the model specification dict for model_key, falling back to DEFAULT_MODEL_KEY."""
    if model_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_key]
    return MODEL_REGISTRY[DEFAULT_MODEL_KEY]
