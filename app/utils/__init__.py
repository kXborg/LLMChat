"""
Utility modules for LLMChat application.
"""

from .rag_utils import (
    get_embedding_model,
    init_qdrant_collection,
    extract_text_from_pdf,
    chunk_text,
    index_document,
    search_documents,
    get_user_documents,
    delete_document,
    perform_rag_search,
    document_metadata,
)

from .vision_utils import (
    probe_vision_capability,
    get_vision_capability,
    get_vision_capability_from_request,
    get_max_history_turns_for_model,
    validate_attachments_for_model,
    VISION_PROBE_CACHE,
)

from .message_utils import (
    inject_attachments_into_message,
    build_user_content,
    normalize_messages_for_vllm,
    build_messages,
)

from .search_utils import (
    perform_web_search,
)

from .image_utils import (
    compress_image,
)

from .llm_utils import (
    parse_allowed_tokens_from_error,
)

__all__ = [
    # RAG utilities
    "get_embedding_model",
    "init_qdrant_collection",
    "extract_text_from_pdf",
    "chunk_text",
    "index_document",
    "search_documents",
    "get_user_documents",
    "delete_document",
    "perform_rag_search",
    "document_metadata",
    # Vision utilities
    "probe_vision_capability",
    "get_vision_capability",
    "get_vision_capability_from_request",
    "get_max_history_turns_for_model",
    "validate_attachments_for_model",
    "VISION_PROBE_CACHE",
    # Message utilities
    "inject_attachments_into_message",
    "build_user_content",
    "normalize_messages_for_vllm",
    "build_messages",
    # Search utilities
    "perform_web_search",
    # Image utilities
    "compress_image",
    # LLM utilities
    "parse_allowed_tokens_from_error",
]
