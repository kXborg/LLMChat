"""
Message building and formatting utilities.
Handles message construction, content normalization, and attachment processing.
"""

import os
import base64
from typing import List, Dict, Any
from urllib.parse import urlparse
from pathlib import Path


def inject_attachments_into_message(base_text: str, attachments: list | None) -> str:
    """
    Inject attachment information into a text message.
    
    Args:
        base_text: Base message text
        attachments: List of attachment objects
        
    Returns:
        Message text with attachment information appended
    """
    if not attachments:
        return base_text
    lines: List[str] = []
    for att in attachments:
        header = f"[Attachment] {att.filename} ({att.mime_type}) URL: {att.url}"
        if att.text:
            # Limit attachment text to avoid huge prompts
            preview = att.text[:5000]
            header += f"\nContent preview:\n{preview}"
        lines.append(header)
    return base_text + "\n\n" + "\n\n".join(lines)


def build_user_content(
    user_message: str,
    attachments: list | None,
    model_id: str,
    upload_dir: Path,
    inline_local_uploads: bool,
    get_vision_capability_func,
    user_id: str = "",
    vision_enabled_override: bool | None = None
) -> str | List[Dict[str, Any]]:
    """
    Build user content with proper formatting for vision or text-only models.
    
    Args:
        user_message: User's message text
        attachments: List of attachments
        model_id: Model identifier
        upload_dir: Directory where uploads are stored
        inline_local_uploads: Whether to inline local uploads as base64
        get_vision_capability_func: Function to check vision capability
        user_id: User identifier
        vision_enabled_override: Optional vision override
        
    Returns:
        Formatted content (string for text-only, list of parts for vision models)
    """
    vision_capable = get_vision_capability_func(user_id, model_id, vision_enabled_override)
    
    if vision_capable:
        parts: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
        for att in attachments or []:
            mt = (att.mime_type or "").lower()
            if mt.startswith("image/"):
                # Ensure absolute URL; client attempts this already, but be tolerant
                url = str(att.url)
                try:
                    # If it's relative, make it absolute to our origin
                    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("data:")):
                        url = url if url.startswith("/") else "/" + url
                        # We cannot know host here reliably; keep as-is and rely on client normalization
                except Exception:
                    pass
                
                # Optionally inline local uploads as data URLs so the model server
                # doesn't need to fetch from our FastAPI host.
                if inline_local_uploads:
                    try:
                        parsed = urlparse(url)
                        path = parsed.path or ""
                        if path.startswith("/uploads/"):
                            fname = os.path.basename(path)
                            fpath = upload_dir / fname
                            if fpath.exists() and fpath.is_file():
                                raw = fpath.read_bytes()
                                b64 = base64.b64encode(raw).decode("ascii")
                                url = f"data:{mt};base64,{b64}"
                    except Exception:
                        # Fall back to original URL if any error occurs
                        pass
                
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
            elif mt.startswith("text/") and att.text:
                parts.append({"type": "text", "text": f"[Attachment {att.filename}]\n{att.text[:5000]}"})
            # PDFs are kept as reference only (no OCR here)
        return parts
    
    # text-only fallback: merge into a single string (with lightweight attachment previews)
    if not attachments:
        return user_message
    lines = [user_message]
    for att in attachments:
        mt = (att.mime_type or "").lower()
        if mt.startswith("text/") and att.text:
            lines.append(f"[Attachment {att.filename}]\n{att.text[:1000]}")
        elif mt.startswith("image/"):
            lines.append(f"[Image reference] {att.url}")
    return "\n\n".join(lines)


def normalize_messages_for_vllm(
    messages: list,
    model_id: str,
    get_vision_capability_func,
    user_id: str = "",
    vision_enabled_override: bool | None = None
) -> list:
    """
    Normalize messages for vLLM compatibility.
    Converts structured content to plain text for text-only models.
    
    Args:
        messages: List of message dictionaries
        model_id: Model identifier
        get_vision_capability_func: Function to check vision capability
        user_id: User identifier
        vision_enabled_override: Optional vision override
        
    Returns:
        Normalized message list
    """
    # If the target model is vision-capable, keep structured parts intact
    vision_capable = get_vision_capability_func(user_id, model_id, vision_enabled_override)
    if vision_capable:
        return messages
    
    normalized = []

    for m in messages:
        content = m.get("content")

        if isinstance(content, str):
            normalized.append(m)
            continue

        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        url = p.get("image_url", {}).get("url", "")
                        if url:
                            parts.append(f"[Image] {url}")

            normalized.append(
                {
                    "role": m.get("role", "user"),
                    "content": "\n".join(parts),
                }
            )
            continue

        normalized.append(
            {
                "role": m.get("role", "user"),
                "content": str(content),
            }
        )

    return normalized


def build_messages(
    user_id: str,
    user_message: str,
    system_prompt: str,
    user_histories: dict,
    attachments: list | None,
    model_id: str,
    upload_dir: Path,
    inline_local_uploads: bool,
    get_max_history_turns_func,
    get_vision_capability_func,
    perform_web_search_func,
    perform_rag_search_func,
    vision_enabled_override: bool | None = None,
    web_search: bool = False,
    rag_enabled: bool = False
) -> list:
    """
    Build complete message list with history, context, and attachments.
    
    Args:
        user_id: User identifier
        user_message: User's message text
        system_prompt: System prompt to use
        user_histories: Dictionary of user conversation histories
        attachments: List of attachments
        model_id: Model identifier
        upload_dir: Directory where uploads are stored
        inline_local_uploads: Whether to inline local uploads
        get_max_history_turns_func: Function to get max history turns
        get_vision_capability_func: Function to check vision capability
        perform_web_search_func: Function to perform web search
        perform_rag_search_func: Function to perform RAG search
        vision_enabled_override: Optional vision override
        web_search: Whether to enable web search
        rag_enabled: Whether to enable RAG
        
    Returns:
        List of message dictionaries ready for the LLM
    """
    history = user_histories.get(user_id)
    if history is None:
        history = [{"role": "system", "content": system_prompt}]

    # Keep system message (first) and only last N turns to reduce context growth
    system_msg = []
    tail = []
    if history and isinstance(history[0], dict) and history[0].get("role") == "system":
        system_msg = [history[0]]
        tail = history[1:]
    else:
        tail = history

    # Each turn is two messages (user+assistant); keep last N turns per model type
    max_tail_msgs = get_max_history_turns_func(model_id, user_id, vision_enabled_override) * 2
    trimmed_tail = tail[-max_tail_msgs:]

    messages = [*system_msg, *trimmed_tail]
    
    # Augment user message with context from various sources
    augmented_message = user_message
    context_parts = []
    
    # Add RAG context if enabled
    if rag_enabled:
        rag_context = perform_rag_search_func(user_id, user_message)
        if rag_context:
            context_parts.append(rag_context)
    
    # Add web search results if enabled
    if web_search:
        search_results = perform_web_search_func(user_message)
        if search_results:
            context_parts.append(f"**[Web Search Context]**\n{search_results}")
    
    # Combine contexts
    if context_parts:
        combined_context = "\n\n---\n".join(context_parts)
        augmented_message = f"{user_message}\n\n---\n{combined_context}\n---\n\nPlease use the above context to help answer my question. Cite sources when relevant."
    
    user_content = build_user_content(
        augmented_message,
        attachments,
        model_id,
        upload_dir,
        inline_local_uploads,
        get_vision_capability_func,
        user_id,
        vision_enabled_override
    )
    messages.append({"role": "user", "content": user_content})
    return messages
