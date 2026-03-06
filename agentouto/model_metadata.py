from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("agentouto")

_API_BASE_URL = "https://lcw-api.blp.sh"

try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False


@dataclass
class ModelMetadata:
    context_window: int
    max_output_tokens: int | None = None


class ModelMetadataError(Exception):
    pass


def _normalize_model_name(name: str) -> str:
    """Normalize model name for consistent cache keys.
    
    gpt-4o, GPT-4O, gpt_4o, gpt 4o → gpt-4o
    """
    normalized = name.lower().strip()
    normalized = re.sub(r'[\s_]+', '-', normalized)
    return normalized


_api_metadata: dict[str, ModelMetadata] = {}


def clear_cache() -> None:
    """Clear the per-model metadata cache."""
    _api_metadata.clear()
    logger.info("Model metadata cache cleared")


async def _fetch_model(model: str) -> ModelMetadata:
    """Fetch metadata for a single model from the LCW API."""
    if not _AIOHTTP_AVAILABLE:
        raise ModelMetadataError(
            "aiohttp not installed. Install with: pip install aiohttp"
        )

    url = f"{_API_BASE_URL}/context-window"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"model": model},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()

                if resp.status == 404 or not data.get("success", False):
                    error_msg = data.get("error", f"Model not found: {model}")
                    raise ModelMetadataError(error_msg)

                if resp.status != 200:
                    raise ModelMetadataError(
                        f"Model metadata API returned status {resp.status}"
                    )

                model_data = data.get("data", {})
                context_window = model_data.get("contextWindow")

                if context_window is None:
                    raise ModelMetadataError(
                        f"No context window data for model '{model}'"
                    )

                meta = ModelMetadata(
                    context_window=context_window,
                    max_output_tokens=context_window,  # Same as context_window
                )

                # Cache under normalized key
                cache_key = _normalize_model_name(model)
                _api_metadata[cache_key] = meta

                # Also cache under slug if available
                slug = model_data.get("slug")
                if slug:
                    _api_metadata[slug.lower()] = meta

                logger.debug(
                    "Fetched metadata for '%s': context_window=%d",
                    model, context_window,
                )

                return meta

    except ModelMetadataError:
        raise
    except Exception as e:
        raise ModelMetadataError(
            f"Failed to fetch model metadata for '{model}': {e}"
        )


async def ensure_loaded() -> None:
    """No-op for backwards compatibility. Models are fetched on demand."""
    pass


async def get_model_info(model: str) -> ModelMetadata:
    """Get model metadata, fetching from API on cache miss."""
    cache_key = _normalize_model_name(model)

    if cache_key in _api_metadata:
        return _api_metadata[cache_key]

    try:
        return await _fetch_model(model)
    except ModelMetadataError:
        # On failure, remove this model's entry from cache if it exists
        _api_metadata.pop(cache_key, None)
        # Also try slug key
        slug_key = model.lower()
        _api_metadata.pop(slug_key, None)
        raise


async def resolve_max_output_tokens(model: str, user_value: int | None) -> int | None:
    """Resolve max output tokens. Returns user value, API value, or None."""
    if user_value is not None:
        return user_value

    try:
        info = await get_model_info(model)
        if info.max_output_tokens:
            return info.max_output_tokens
    except ModelMetadataError:
        logger.debug(
            "Could not resolve max_output_tokens for '%s', returning None",
            model,
        )

    return None


async def get_context_window(model: str) -> int:
    """Get context window size for a model."""
    info = await get_model_info(model)
    return info.context_window
