"""Model factory. Teams ask for a chat model; the platform decides which one.

``AGENTDOCK_MODEL``:
- ``offline`` (default): a scripted model driven by the agent's own ``offline_policy``.
  Free, deterministic, no keys, so CI and evals always run.
- ``cloudflare:<model>``: Cloudflare Workers AI through its OpenAI-compatible
  endpoint (free daily allowance), e.g. ``cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast``.
  Needs ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_AI_TOKEN``.
- ``gemini:<model>``: Google Gemini via its native API (free tier), e.g.
  ``gemini:gemini-3.5-flash-lite``. Needs ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

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
        return ChatResult(generations=[ChatGeneration(message=msg)])


def model_name() -> str:
    return os.environ.get("AGENTDOCK_MODEL", "offline")


def chat_model(offline_policy: OfflinePolicy, *, temperature: float = 0.0) -> BaseChatModel:
    name = model_name()
    if name == "offline":
        return ScriptedChatModel(policy=offline_policy)
    if name.startswith("cloudflare:"):
        from langchain_openai import ChatOpenAI

        account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        return ChatOpenAI(
            model=name.split(":", 1)[1],
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
            api_key=os.environ["CLOUDFLARE_AI_TOKEN"],
            temperature=temperature,
            timeout=30,
            max_retries=2,
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
            timeout=30,
            max_retries=2,
        )
    raise ValueError(f"Unknown AGENTDOCK_MODEL {name!r} (use 'offline', 'cloudflare:<model>' or 'gemini:<model>')")
