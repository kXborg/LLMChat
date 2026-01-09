"""
Vision and model capability detection utilities.
Handles vision capability probing, validation, and configuration.
"""

import requests
from fastapi import HTTPException


# Cache for vision capability probe results
VISION_PROBE_CACHE = {}


def probe_vision_capability(base_url: str, model_id: str, timeout: int = 5) -> bool:
    """
    Probe whether a model supports vision/multimodal inputs.
    
    Args:
        base_url: Base URL of the OpenAI-compatible API
        model_id: Model identifier to probe
        timeout: Request timeout in seconds
        
    Returns:
        True if model supports vision, False otherwise
    """
    if model_id in VISION_PROBE_CACHE:
        return VISION_PROBE_CACHE[model_id]

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "about:blank"},
                    {"type": "text", "text": "ping"},
                ],
            }
        ],
        "max_tokens": 1,
    }

    # Normalize the base URL - strip trailing /v1 if present, then add it back
    # This ensures consistency whether base_url ends with /v1 or not
    normalized_base = base_url.rstrip("/")
    if normalized_base.endswith("/v1"):
        normalized_base = normalized_base[:-3]
    
    try:
        r = requests.post(
            f"{normalized_base}/v1/chat/completions",
            json=payload,
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=timeout,
        )

        if r.status_code == 200:
            VISION_PROBE_CACHE[model_id] = True
            return True

        error_text = r.text.lower()

        if any(k in error_text for k in [
            "image",
            "vision",
            "multimodal",
            "image_url",
            "image token",
        ]):
            VISION_PROBE_CACHE[model_id] = True
            return True

        VISION_PROBE_CACHE[model_id] = False
        return False

    except requests.RequestException:
        VISION_PROBE_CACHE[model_id] = False
        return False


def get_vision_capability(
    base_url: str,
    user_id: str,
    model_id: str,
    vision_capability_overrides: dict
) -> bool:
    """
    Get vision capability for a model, checking for user override first.
    If user has set an override for this model, use that.
    Otherwise, use the probed capability.
    
    Args:
        base_url: Base URL of the OpenAI-compatible API
        user_id: User identifier
        model_id: Model identifier
        vision_capability_overrides: Dictionary of user overrides
        
    Returns:
        True if vision is enabled, False otherwise
    """
    override_key = f"{user_id}:{model_id}"
    override = vision_capability_overrides.get(override_key)
    if override is not None:
        return override
    return probe_vision_capability(base_url, model_id)


def get_vision_capability_from_request(
    base_url: str,
    user_id: str,
    model_id: str,
    vision_capability_overrides: dict,
    vision_enabled_override: bool | None = None
) -> bool:
    """
    Get vision capability, checking user's request override first.
    Priority: request override > stored override > probed capability
    
    Args:
        base_url: Base URL of the OpenAI-compatible API
        user_id: User identifier
        model_id: Model identifier
        vision_capability_overrides: Dictionary of user overrides
        vision_enabled_override: Optional override from the request
        
    Returns:
        True if vision is enabled, False otherwise
    """
    if vision_enabled_override is not None:
        return vision_enabled_override
    return get_vision_capability(base_url, user_id, model_id, vision_capability_overrides)


def get_max_history_turns_for_model(
    base_url: str,
    model_id: str,
    max_history_turns_text: int,
    max_history_turns_vision: int,
    vision_capability_overrides: dict,
    user_id: str = "",
    vision_enabled_override: bool | None = None
) -> int:
    """
    Get maximum history turns based on model's vision capability.
    
    Args:
        base_url: Base URL of the OpenAI-compatible API
        model_id: Model identifier
        max_history_turns_text: Max turns for text-only models
        max_history_turns_vision: Max turns for vision models
        vision_capability_overrides: Dictionary of user overrides
        user_id: User identifier
        vision_enabled_override: Optional override from the request
        
    Returns:
        Maximum number of history turns
    """
    vision_capable = get_vision_capability_from_request(
        base_url, user_id, model_id, vision_capability_overrides, vision_enabled_override
    )
    return max_history_turns_vision if vision_capable else max_history_turns_text


def validate_attachments_for_model(
    base_url: str,
    model_id: str,
    attachments: list | None,
    vision_capability_overrides: dict,
    user_id: str = "",
    vision_enabled_override: bool | None = None
):
    """
    Validate that attachments are compatible with the model.
    Raises HTTPException if non-text attachments are sent to text-only model.
    
    Args:
        base_url: Base URL of the OpenAI-compatible API
        model_id: Model identifier
        attachments: List of attachments
        vision_capability_overrides: Dictionary of user overrides
        user_id: User identifier
        vision_enabled_override: Optional override from the request
        
    Raises:
        HTTPException: If attachments are incompatible with model
    """
    if not attachments:
        return
    vision_capable = get_vision_capability_from_request(
        base_url, user_id, model_id, vision_capability_overrides, vision_enabled_override
    )
    if vision_capable:
        return
    for att in attachments:
        mt = (att.mime_type or "").lower()
        # Only allow text/* for text-only models
        if not mt.startswith("text/"):
            raise HTTPException(
                status_code=400,
                detail="Selected model is text-only; remove non-text attachments (images/PDFs)."
            )
