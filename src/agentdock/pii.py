"""PII redaction applied by the platform on the way in and on the way out.

Agents don't have to remember to scrub anything: the wrapper redacts user input
before the agent sees it and redacts every message the agent produces before
it is stored or returned (REST, MCP and A2A all read the same state).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PATTERNS: dict[str, re.Pattern[str]] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"(?<!\w)\+?\d{1,3}[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
}


@dataclass
class Redaction:
    text: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> Redaction:
    counts: dict[str, int] = {}
    for label, pattern in PATTERNS.items():
        text, n = pattern.subn(f"[{label} REDACTED]", text)
        if n:
            counts[label] = n
    return Redaction(text, counts)


def redact_content(content: object) -> tuple[object, dict[str, int]]:
    """Redact a message's content, which may be a string or a list of content blocks."""
    if isinstance(content, str):
        r = redact(content)
        return r.text, r.counts
    if isinstance(content, list):
        out, counts = [], {}
        for block in content:
            if isinstance(block, str):
                r = redact(block)
                out.append(r.text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                r = redact(block["text"])
                out.append({**block, "text": r.text})
            else:
                out.append(block)
                continue
            for k, v in r.counts.items():
                counts[k] = counts.get(k, 0) + v
        return out, counts
    return content, {}
