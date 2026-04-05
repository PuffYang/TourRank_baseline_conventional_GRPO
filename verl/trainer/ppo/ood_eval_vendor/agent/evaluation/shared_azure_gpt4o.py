from __future__ import annotations

import os
from typing import Any

from openai import AsyncAzureOpenAI, AzureOpenAI

DEFAULT_AZURE_OPENAI_API_VERSION = "2024-06-01"
DEFAULT_AZURE_OPENAI_ENDPOINT = (
    "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-06-01"
)
DEFAULT_AZURE_GPT4O_MODEL = "gpt-4o"


def resolve_azure_gpt4o_model(requested_model: str | None = None) -> str:
    env_model = os.environ.get("AZURE_OPENAI_MODEL")
    if env_model:
        return env_model

    if requested_model:
        normalized = str(requested_model).strip()
        if normalized.startswith("openai/"):
            normalized = normalized.split("/", 1)[1]
        if normalized.lower() in {"gpt-4o", "gpt-4-o"}:
            return normalized

    return DEFAULT_AZURE_GPT4O_MODEL


def get_azure_openai_config() -> dict[str, str]:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Azure GPT-4o API key. Set AZURE_OPENAI_API_KEY (preferred) or OPENAI_API_KEY."
        )

    return {
        "api_key": api_key,
        "api_version": os.environ.get("OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION),
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_OPENAI_ENDPOINT),
    }


def get_azure_openai_client() -> AzureOpenAI:
    return AzureOpenAI(**get_azure_openai_config())


def get_async_azure_openai_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(**get_azure_openai_config())


def create_chat_completion(
    *,
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int | float = 200,
    client: AzureOpenAI | None = None,
):
    active_client = client or get_azure_openai_client()
    return active_client.chat.completions.create(
        model=resolve_azure_gpt4o_model(model),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def create_chat_completion_text(
    *,
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    timeout: int | float = 200,
    client: AzureOpenAI | None = None,
) -> tuple[str, Any]:
    response = create_chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        client=client,
    )
    content = response.choices[0].message.content or ""
    return content, response
