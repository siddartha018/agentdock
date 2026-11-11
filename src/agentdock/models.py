"""Model factory. Teams ask for a chat model; the platform decides which one.

``AGENTDOCK_MODEL``:
- ``offline`` (default): a scripted model driven by the agent's own ``offline_policy``.
  Free, deterministic, no keys, so CI and evals always run.
- ``cloudflare:<model>``: Cloudflare Workers AI through its OpenAI-compatible
  endpoint (free daily allowance), e.g. ``cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast``.
  Needs ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_AI_TOKEN``.
- ``AGENTDOCK_FALLBACK=offline`` (with a real model): if a call fails (rate limit,
  outage, timeout) the agent's offline policy answers instead, so a public demo
  degrades instead of erroring. Answers are tagged with the model that produced them.
- ``gemini:<model>``: Google Gemini via its native API (free tier), e.g.
  ``gemini:gemini-3.5-flash-lite``. Needs ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
import json
import re
from typing import Iterator

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

OFFLINE_TAG = "agentdock-offline"

# (messages, tool names) -> the next AI message (text and/or tool calls)
OfflinePolicy = Callable[[Sequence[BaseMessage], list[str]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """A chat model whose behaviour is a plain Python function.

    It supports ``bind_tools`` so the same ReAct graph runs unchanged with the
    scripted policy or a real model.
    """

    policy: Any
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "agentdock-offline"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        names = []
        for t in tools:
            if isinstance(t, BaseTool):
                names.append(t.name)
            else:
                names.append(convert_to_openai_tool(t)["function"]["name"])
        return self.model_copy(update={"tool_names": names})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg = self.policy(messages, self.tool_names)
        msg.response_metadata = {**msg.response_metadata, "model_name": OFFLINE_TAG}
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream word by word like a real model, so the streaming path is tested offline."""
        msg = self.policy(messages, self.tool_names)
        text = msg.content if isinstance(msg.content, str) else ""
        for i, piece in enumerate(re.findall(r"\S+\s*|\s+", text)):
            # metadata strings concatenate when chunks merge, so tag only the first
            meta = {"model_name": OFFLINE_TAG} if i == 0 else {}
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece, response_metadata=meta))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk
        if msg.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    response_metadata={} if text else {"model_name": OFFLINE_TAG},
                    tool_call_chunks=[
                        {"name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": i}
                        for i, tc in enumerate(msg.tool_calls)
                    ],
                )
            )


def model_name() -> str:
    return os.environ.get("AGENTDOCK_MODEL", "offline")


class WithOfflineFallback:
    """A real model first; the agent's scripted policy if the call fails.

    ``bind_tools`` binds both, so the ReAct graph code is unchanged.
    """

    def __init__(self, primary: BaseChatModel, offline: ScriptedChatModel) -> None:
        self.primary, self.offline = primary, offline

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self.primary.bind_tools(tools, **kwargs).with_fallbacks([self.offline.bind_tools(tools)])

    def invoke(self, *a: Any, **k: Any) -> Any:
        return self.primary.with_fallbacks([self.offline]).invoke(*a, **k)

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return await self.primary.with_fallbacks([self.offline]).ainvoke(*a, **k)


def model_used(message: Any) -> str:
    """Which model wrote this message: 'offline' or the provider's model name."""
    name = (getattr(message, "response_metadata", None) or {}).get("model_name") or ""
    return "offline" if not name or OFFLINE_TAG in str(name) else str(name)


def chat_model(offline_policy: OfflinePolicy, *, temperature: float = 0.0) -> Any:
    name = model_name()
    if name == "offline":
        return ScriptedChatModel(policy=offline_policy)
    fallback = os.environ.get("AGENTDOCK_FALLBACK") == "offline"
    try:
        real = _real_model(name, temperature, fallback=fallback)
    except KeyError as missing:  # the model's API key isn't set
        if not fallback:
            raise
        print(f"[agentdock] {name}: {missing} not set, using the offline model", flush=True)
        return ScriptedChatModel(policy=offline_policy)
    if fallback:
        return WithOfflineFallback(real, ScriptedChatModel(policy=offline_policy))
    return real


def _real_model(name: str, temperature: float, fallback: bool) -> BaseChatModel:
    retries = 0 if fallback else 2  # with a fallback, fail fast instead of retrying a rate limit
    if name.startswith("cloudflare:"):
        from langchain_openai import ChatOpenAI

        account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        return ChatOpenAI(
            model=name.split(":", 1)[1],
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
            api_key=os.environ["CLOUDFLARE_AI_TOKEN"],
            temperature=temperature,
            timeout=30,
            max_retries=retries,
        )
    if name.startswith("gemini:"):
        # Native client, not the OpenAI-compatible endpoint: Gemini 3 tool calls
        # carry thought signatures that must be echoed back on the next turn,
        # and the OpenAI client drops them (HTTP 400 on the second call).
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=name.split(":", 1)[1],
            google_api_key=os.environ["GEMINI_API_KEY"],
            temperature=temperature,
            timeout=20 if fallback else 30,
            max_retries=retries,
        )
    raise ValueError(f"Unknown AGENTDOCK_MODEL {name!r} (use 'offline', 'cloudflare:<model>' or 'gemini:<model>')")
