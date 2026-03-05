from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("agentouto")

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False


@dataclass
class ModelMetadata:
    context_window: int
    max_output_tokens: int | None = None


_MINIMAL_FALLBACK: dict[str, ModelMetadata] = {
    "gpt-5": ModelMetadata(context_window=400_000, max_output_tokens=128_000),
    "gpt-5-mini": ModelMetadata(context_window=400_000, max_output_tokens=128_000),
    "gpt-5-nano": ModelMetadata(context_window=400_000, max_output_tokens=128_000),
    "gpt-4o": ModelMetadata(context_window=128_000, max_output_tokens=16_384),
    "claude-opus-4-6": ModelMetadata(context_window=200_000, max_output_tokens=32_768),
    "claude-sonnet-4-6": ModelMetadata(context_window=200_000, max_output_tokens=32_768),
    "gemini-2.5-pro": ModelMetadata(context_window=1_000_000, max_output_tokens=64_000),
}


_loaded: bool = False
_api_metadata: dict[str, ModelMetadata] = {}


async def _load_from_openrouter() -> dict[str, ModelMetadata]:
    global _loaded, _api_metadata
    if _loaded:
        return _api_metadata
    
    if not _AIOHTTP_AVAILABLE:
        _loaded = True
        return {}
    
    url = "https://openrouter.ai/api/v1/models"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for model_entry in data.get("data", []):
                        model_id = model_entry.get("id", "").lower()
                        endpoints = model_entry.get("endpoints", [])
                        if not endpoints:
                            continue
                        endpoint = endpoints[0]
                        context_length = endpoint.get("context_length")
                        if not context_length:
                            continue
                        max_output = endpoint.get("max_output_tokens")
                        meta = ModelMetadata(
                            context_window=context_length,
                            max_output_tokens=max_output,
                        )
                        _api_metadata[model_id] = meta
                        
                        author = model_entry.get("author", "")
                        name = model_entry.get("name", "").lower()
                        if author and name:
                            short_key = f"{author}/{name}"
                            _api_metadata[short_key] = meta
                    
                    logger.info(
                        "Loaded %d models from OpenRouter API",
                        len(_api_metadata),
                    )
                else:
                    logger.warning(
                        "OpenRouter API returned status %s", resp.status
                    )
    except Exception as e:
        logger.warning("Failed to fetch model metadata from OpenRouter: %s", e)
    
    _loaded = True
    return _api_metadata


async def ensure_loaded() -> None:
    await _load_from_openrouter()


def get_model_info(model: str) -> ModelMetadata | None:
    model_lower = model.lower()
    
    if model_lower in _api_metadata:
        return _api_metadata[model_lower]
    
    if model_lower in _MINIMAL_FALLBACK:
        return _MINIMAL_FALLBACK[model_lower]
    
    for known, meta in _api_metadata.items():
        if known in model_lower or model_lower in known:
            return meta
    
    for known, meta in _MINIMAL_FALLBACK.items():
        if known in model_lower or model_lower in known:
            return meta
    
    return None


def resolve_max_output_tokens(model: str, user_value: int | None) -> int | None:
    if user_value is not None:
        return user_value
    
    info = get_model_info(model)
    if info and info.max_output_tokens:
        return info.max_output_tokens
    
    return None


async def get_context_window(model: str) -> int | None:
    await ensure_loaded()
    info = get_model_info(model)
    if info:
        return info.context_window
    return None
