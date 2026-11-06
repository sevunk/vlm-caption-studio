"""
core/text_cleaner.py
Anti-leak & Anti-truncation VLM output cleaner across all 7 VLM architectures:
1. Strips closed `<think>...</think>` blocks.
2. For captioning (`allow_reasoning_fallback=False`), never leaks unfinished `<think>` fragments; returns `""` if `<think>` was unclosed so caller can auto-retry with higher token budget.
3. Strips ANSI escape sequences, special chat template tokens (`<|eot_id|>`, `<end_of_turn>`, `</s>`), and conversational meta-preambles.
4. Strips trailing baked-in disclaimers and cleans unfinished dangling tails if cut off.
5. Normalizes trigger words, commas, and provides word/token estimation.
"""

import re
from typing import Any, Dict, Optional

META_PREAMBLE_PATTERNS = [
    r"^(?:here\s+is|here's)\s+(?:a|the|an|your)\s+(?:detailed\s+|uncensored\s+|concise\s+|fluent\s+|single-paragraph\s+)?(?:caption|description|prompt|tag\s+list)(?:\s+for\s+(?:the|this)\s+image)?\s*[:\-–—]\s*",
    r"^(?:sure|certainly|of\s+course)[,!\.]?\s+(?:here\s+is|here's)\s+[^:\n]{1,60}:\s*",
    r"^(?:caption|description|prompt)\s*[:\-–—]\s*",
]

TRAILING_DISCLAIMER_PATTERNS = [
    r"\s*(?:Note|Disclaimer|Please note)\s*:\s*This\s+is\s+(?:general\s+information|a\s+fictional|an\s+artistic).*$",
]


def _extract_last_descriptive_paragraph(reasoning_text: str) -> str:
    """
    Used ONLY in chat mode (`allow_reasoning_fallback=True`) if a model puts its entire reply
    inside `reasoning_content`. Never used for dataset captioning to prevent mid-thought fragments.
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


def extract_message_text(message_obj: Dict[str, Any], allow_reasoning_fallback: bool = False) -> str:
    """
    Safely extract text from an OpenAI-compatible `/v1/chat/completions` message object.
    - When `allow_reasoning_fallback=False` (dataset captioning), if `content` only contains an unclosed
      `<think>` block (meaning thinking exhausted `max_tokens`), returns `""` so the caller retries cleanly
      rather than saving a broken mid-sentence fragment.
    """
    if not isinstance(message_obj, dict):
        return ""
    content = (message_obj.get("content") or "").strip()
    reasoning = (message_obj.get("reasoning_content") or "").strip()

    if content:
        # Remove closed <think>...</think> blocks first
        without_closed = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        # Check if there is still an unclosed <think> block
        if re.search(r"<think>", without_closed, flags=re.IGNORECASE):
            before_unclosed = re.sub(r"<think>.*$", "", without_closed, flags=re.DOTALL | re.IGNORECASE).strip()
            if before_unclosed:
                return before_unclosed
            if allow_reasoning_fallback:
                inner = re.sub(r"^.*?<think>", "", without_closed, flags=re.DOTALL | re.IGNORECASE)
                return _extract_last_descriptive_paragraph(inner)
            return ""
        if without_closed:
            return without_closed

    if not content and reasoning and allow_reasoning_fallback:
        return _extract_last_descriptive_paragraph(reasoning)

    return ""


def clean_vlm_response(
    text: str,
    sys_prompt: str = "",
    usr_prompt: str = "",
    strip_meta_preamble: bool = True,
    single_line: bool = True,
    repair_dangling_sentence: bool = False,
) -> str:
    """
    Clean raw VLM output from llama-server or llama-mtmd-cli:
    - Removes closed <think>...</think> and unclosed <think>...$
    - Removes ANSI escape codes and <|...|>, </s>, <end_of_turn>, <|eot_id|> tokens
    - Strips CLI prompt echo and conversational meta preambles
    - Optionally trims dangling incomplete sentence tails if a model hit a hard token wall
    """
    if not text:
        return ""

    # 1. Remove closed <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Remove unclosed/truncated <think>...$ blocks (never leak raw thinking into captions)
    if re.search(r"<think>", text, flags=re.IGNORECASE):
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # 3. Also strip stray </think> if model emitted closing tag without opening tag
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]

    # 4. Strip ANSI escape sequences and special chat tokens across Qwen, Llama-3, and Gemma-3
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"</?(?:s|bos|eos|end_of_turn|start_of_turn|eot_id)>", "", text, flags=re.IGNORECASE)

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
        for dpat in TRAILING_DISCLAIMER_PATTERNS:
            text = re.sub(dpat, "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    if single_line:
        text = " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()
    else:
        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines).strip()

    # Unwrap outer quotes if the model wrapped the entire paragraph in quotes
    if len(text) >= 2 and ((text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'"))):
        text = text[1:-1].strip()

    if repair_dangling_sentence and text:
        text = _repair_unfinished_prose(text)

    return text


def _repair_unfinished_prose(text: str) -> str:
    """
    If a natural-language paragraph caption (not a comma-separated booru tag list) was cut off
    mid-sentence without terminal punctuation, trim back to the last complete sentence if sufficient
    content exists, or cleanly strip trailing prepositions/articles/conjunctions and add a period.
    """
    s = text.strip()
    if not s:
        return s

    # Detect comma-separated booru tag list (many commas, no periods) -> do not alter as prose
    words = s.split()
    comma_count = s.count(",")
    if "." not in s and comma_count >= max(3, len(words) // 3):
        return s.rstrip(", ")

    # Already ends with valid terminal punctuation
    if s[-1] in '.!?"\')]}':
        return s

    # If there are multiple sentences and the last tail is short (< 18 words), trim to the last period
    last_dot = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
    if last_dot > len(s) * 0.45:
        tail = s[last_dot + 1 :].strip()
        if len(tail.split()) <= 18:
            return s[: last_dot + 1].strip()

    # Otherwise remove dangling conjunctions/prepositions/articles at the very end and close with a period
    s = re.sub(
        r"\s+\b(?:and|or|with|without|in|on|at|to|from|by|of|for|as|into|through|while|where|that|which|who|whose|its|their|his|her|a|an|the)\s*$",
        "",
        s,
        flags=re.IGNORECASE,
    ).rstrip(",;:-–— ")
    if s and s[-1] not in '.!?"\')]}':
        s += "."
    return s


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
