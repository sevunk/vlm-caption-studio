"""
core/text_cleaner.py
Anti-leak VLM output cleaner:
1. Handles empty `content` with populated `reasoning_content` fallback.
2. Strips closed `<think>...</think>` blocks.
3. Strips unclosed/truncated `<think>...$` blocks when cut off by max_tokens.
4. Strips ANSI escape sequences, special chat template tokens, and conversational meta-preambles.
5. Normalizes trigger words, commas, and provides word/token estimation.
"""

import re
from typing import Any, Dict, Optional

META_PREAMBLE_PATTERNS = [
    r"^(?:here\s+is|here's)\s+(?:a|the|an|your)\s+(?:detailed\s+|uncensored\s+|concise\s+|fluent\s+)?(?:caption|description|prompt|tag\s+list)(?:\s+for\s+(?:the|this)\s+image)?\s*[:\-–—]\s*",
    r"^(?:sure|certainly|of\s+course)[,!\.]?\s+(?:here\s+is|here's)\s+[^:\n]{1,60}:\s*",
    r"^(?:caption|description|prompt)\s*[:\-–—]\s*",
]


def _extract_last_descriptive_paragraph(reasoning_text: str) -> str:
    """
    Fallback extractor when a hybrid thinking model puts its response inside reasoning_content
    and leaves content empty. Extracts the last substantial non-meta paragraph/sentence.
    """
    if not reasoning_text:
        return ""
    cleaned = re.sub(r"<think>|</think>", "", reasoning_text, flags=re.IGNORECASE).strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paragraphs:
        return ""
    candidate = paragraphs[-1]
    candidate = re.sub(r'^["\']|["\']$', "", candidate).strip()
    return candidate


def extract_message_text(message_obj: Dict[str, Any], allow_reasoning_fallback: bool = True) -> str:
    """
    Safely extract text from an OpenAI-compatible `/v1/chat/completions` message object.
    Handles both standard `content` and rare `reasoning_content` placement.
    """
    if not isinstance(message_obj, dict):
        return ""
    content = (message_obj.get("content") or "").strip()
    reasoning = (message_obj.get("reasoning_content") or "").strip()

    if content:
        # Check if content only contains an unclosed <think> block and nothing else
        stripped_think = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        stripped_think = re.sub(r"<think>.*$", "", stripped_think, flags=re.DOTALL | re.IGNORECASE).strip()
        if stripped_think:
            return content
        if allow_reasoning_fallback:
            inner = re.sub(r"^.*?<think>", "", content, flags=re.DOTALL | re.IGNORECASE)
            return _extract_last_descriptive_paragraph(inner)

    if not content and reasoning and allow_reasoning_fallback:
        return _extract_last_descriptive_paragraph(reasoning)

    return content


def clean_vlm_response(
    text: str,
    sys_prompt: str = "",
    usr_prompt: str = "",
    strip_meta_preamble: bool = True,
    single_line: bool = True,
) -> str:
    """
    Clean raw VLM output from llama-server or llama-mtmd-cli:
    - Removes closed <think>...</think> and unclosed <think>...$
    - Removes ANSI escape codes and <|...|>, </s>, <end_of_turn> tokens
    - Strips CLI prompt echo if present
    - Optionally removes meta preambles ("Here is a caption for the image:")
    """
    if not text:
        return ""

    # 1. Remove closed <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Remove unclosed/truncated <think>...$ blocks (when cut off by max_tokens)
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        before_think = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if before_think:
            text = before_think
        else:
            # If the entire text started with <think> and never closed, extract last paragraph
            raw_inner = re.sub(r"^.*?<think>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = _extract_last_descriptive_paragraph(raw_inner)

    # 3. Also strip stray </think> if model emitted closing tag without opening tag
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]

    # 4. Strip ANSI escape sequences and special tokens
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"</?(?:s|bos|eos|end_of_turn|start_of_turn)>", "", text, flags=re.IGNORECASE)

    # 5. Strip CLI prompt echoes if any
    if usr_prompt and f"> {usr_prompt}" in text:
        text = text.split(f"> {usr_prompt}")[-1]
    text = re.split(r"\[\s*Prompt:", text, flags=re.IGNORECASE)[0]
    text = re.split(r"\bllama_perf_", text, flags=re.IGNORECASE)[0]
    text = re.split(r"\bExiting\.\.\.", text, flags=re.IGNORECASE)[0]

    text = text.strip()

    # 6. Strip conversational meta preambles for direct captioning
    if strip_meta_preamble:
        for pat in META_PREAMBLE_PATTERNS:
            text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    if single_line:
        text = " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()
    else:
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines).strip()

    return text


def process_caption_text(
    text: str,
    trigger: str,
    pos: str = "front",
    force_lower: bool = False,
    skip_dup: bool = True,
    fix_commas: bool = True,
) -> Optional[str]:
    """
    Insert a trigger word at the front or back of caption text, optionally normalizing case
    and cleaning redundant commas. Returns None if skipped due to duplicate detection.
    """
    text = (text or "").strip()
    trigger = (trigger or "").strip()

    if force_lower:
        text = text.lower()
        trigger = trigger.lower()

    if not trigger:
        if fix_commas and text:
            text = _normalize_commas(text)
        return text

    if skip_dup and trigger.lower() in text.lower():
        return None

    if not text:
        return trigger

    if pos == "front":
        cleaned_body = re.sub(r"^[\s,\.\-]+", "", text)
        new_text = f"{trigger}, {cleaned_body}"
    else:
        has_dot = text.endswith(".")
        cleaned_body = re.sub(r"[\s,\.\-]+$", "", text)
        new_text = f"{cleaned_body}, {trigger}"
        if has_dot:
            new_text += "."

    if fix_commas:
        new_text = _normalize_commas(new_text)

    return new_text.strip()


def _normalize_commas(text: str) -> str:
    """Clean duplicate commas and spacing around commas."""
    text = re.sub(r",+", ",", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\s*,\s*", "", text)
    text = re.sub(r"\s*,\s*$", "", text)
    return text.strip()


def estimate_token_count(text: str) -> Dict[str, int]:
    """
    Calculate word count and estimated subword token count (~1.3x words + punctuation).
    Useful for Diffusion/LoRA token budget indicators (75 / 150 / 225 tokens).
    """
    s = (text or "").strip()
    if not s:
        return {"words": 0, "tokens": 0}
    words = len(s.split())
    comma_count = s.count(",")
    est_tokens = max(words, int(round(words * 1.28 + comma_count * 0.35)))
    return {"words": words, "tokens": est_tokens}
