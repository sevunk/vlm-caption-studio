"""Configuration package for Vision AI Studio."""
from .vlm_catalog import (
    MODEL_REGISTRY,
    STYLE_PRESETS,
    DEFAULT_MODEL_KEY,
    PIXAI_TAGGER_KEY,
    get_model_spec,
)

__all__ = [
    "MODEL_REGISTRY",
    "STYLE_PRESETS",
    "DEFAULT_MODEL_KEY",
    "PIXAI_TAGGER_KEY",
    "get_model_spec",
]
