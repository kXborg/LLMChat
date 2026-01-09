"""
LLM-related utility functions.
Handles token parsing and LLM-specific operations.
"""

import re


def parse_allowed_tokens_from_error(msg: str) -> int | None:
    """
    Parse the allowed token count from an LLM error message.
    
    Typical format:
    "This model's maximum context length is 1024 tokens and your request has 899 input tokens (256 > 1024 - 899)."
    
    Args:
        msg: Error message from the LLM
        
    Returns:
        Number of allowed tokens, or None if parsing fails
    """
    try:
        max_ctx_match = re.search(r"maximum context length is (\d+) tokens", msg)
        input_match = re.search(r"your request has (\d+) input tokens", msg)
        if not max_ctx_match or not input_match:
            return None
        max_ctx = int(max_ctx_match.group(1))
        input_tokens = int(input_match.group(1))
        # Use a small margin for safety
        context_margin = 16
        allowed = max_ctx - input_tokens - context_margin
        return allowed if allowed > 0 else 0
    except Exception:
        return None
