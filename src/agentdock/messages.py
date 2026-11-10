"""Message helpers shared by the wrapper, evals and A2A loopback."""

from __future__ import annotations

from typing import Any


def text_of(msg_or_content: Any) -> str:
    """Plain text of a message or its content, whatever shape the model returned.

    Some providers (Gemini, Anthropic) return a list of content blocks, not a string.
    """
    content = getattr(msg_or_content, "content", msg_or_content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return "" if content is None else str(content)
