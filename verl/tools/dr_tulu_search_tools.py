import asyncio
import json
import os
from typing import Any, Optional
from uuid import uuid4

import requests

from .base_tool import BaseTool
from .schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema
from .schemas import OpenAIFunctionToolSchema, ToolResponse

DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set")
    return value


class _DrTuluBaseTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema)
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"call_index": 0}
        return instance_id, ToolResponse()

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

    def _next_result_id(self, instance_id: str, prefix: str) -> str:
        state = self._instance_dict.setdefault(instance_id, {"call_index": 0})
        state["call_index"] += 1
        return f"{prefix}_{instance_id[:8]}_{state['call_index']}"

    @staticmethod
    def _wrap_tool_output(content_blocks: list[str]) -> str:
        content = "\n".join(block for block in content_blocks if block)
        return f"<tool_output>\n{content}\n</tool_output>"

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()


class GoogleSearchTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="google_search",
                description="General web search for relevant webpages and snippets.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The search query.",
                        ),
                        "gl": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Geolocation country code, e.g. us.",
                        ),
                        "hl": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Host language code, e.g. en.",
                        ),
                        "num_results": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Maximum number of results to return.",
                        ),
                    },
                    required=["query"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = self._clean_text(parameters.get("query"))
        if not query:
            return ToolResponse(text="Error: query is required."), 0.0, {"status": "error"}

        def _search():
            response = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": _require_env_var("SERPER_API_KEY"),
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "q": query,
                        "num": int(parameters.get("num_results", self.config.get("num_results", 5))),
                        "gl": parameters.get("gl", self.config.get("gl", "us")),
                        "hl": parameters.get("hl", self.config.get("hl", "en")),
                        "type": "search",
                    }
                ),
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        try:
            payload = await asyncio.to_thread(_search)
            organic_results = payload.get("organic", []) or []
            blocks = []
            for result in organic_results:
                result_id = self._next_result_id(instance_id, "snippet")
                lines = [
                    f"Title: {self._clean_text(result.get('title'))}",
                    f"URL: {self._clean_text(result.get('link'))}",
                    f"Search Snippet: {self._clean_text(result.get('snippet'))}",
                ]
                if result.get("date"):
                    lines.append(f"Date: {self._clean_text(result.get('date'))}")
                blocks.append(f"<snippet id={result_id}>\n" + "\n".join(lines) + "\n</snippet>")
            if not blocks:
                empty_id = self._next_result_id(instance_id, "snippet")
                blocks.append(f"<snippet id={empty_id}>\nNo search results found.\n</snippet>")
            return (
                ToolResponse(text=self._wrap_tool_output(blocks)),
                0.0,
                {"status": "success", "total_results": len(organic_results), "query_count": 1},
            )
        except Exception as e:
            return ToolResponse(text=f"Error performing google_search: {e}"), 0.0, {"status": "error", "error": str(e)}


class SnippetSearchTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="snippet_search",
                description="Focused snippet retrieval from scientific papers using Semantic Scholar.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The paper or topic query.",
                        ),
                        "limit": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Number of snippets to retrieve.",
                        ),
                        "year": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Publication year or year range, e.g. 2022-2025.",
                        ),
                        "fieldsOfStudy": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Optional fields of study hint. Unsupported values are ignored.",
                        ),
                        "venue": OpenAIFunctionPropertySchema(
                            type="string",
                            description="Optional venue filter, e.g. ACL.",
                        ),
                    },
                    required=["query"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = self._clean_text(parameters.get("query"))
        if not query:
            return ToolResponse(text="Error: query is required."), 0.0, {"status": "error"}

        def _search():
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/snippet/search",
                params={
                    "query": query,
                    "limit": int(parameters.get("limit", self.config.get("limit", 5))),
                    **({"year": parameters["year"]} if parameters.get("year") else {}),
                    **({"venue": parameters["venue"]} if parameters.get("venue") else {}),
                },
                headers={"x-api-key": _require_env_var("S2_API_KEY")},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        try:
            payload = await asyncio.to_thread(_search)
            snippets = payload.get("data", []) or []
            blocks = []
            for item in snippets:
                result_id = self._next_result_id(instance_id, "snippet")
                paper = item.get("paper", {}) or {}
                snippet_info = item.get("snippet", {}) or {}
                lines = [
                    f"Title: {self._clean_text(paper.get('title'))}",
                    f"Paper ID: {self._clean_text(paper.get('corpusId'))}",
                    f"Snippet: {self._clean_text(snippet_info.get('text'))}",
                ]
                if paper.get("authors"):
                    lines.append(f"Authors: {', '.join(str(author) for author in paper.get('authors', []))}")
                blocks.append(f"<snippet id={result_id}>\n" + "\n".join(lines) + "\n</snippet>")
            if not blocks:
                empty_id = self._next_result_id(instance_id, "snippet")
                blocks.append(f"<snippet id={empty_id}>\nNo snippet search results found.\n</snippet>")
            metrics = {
                "status": "success",
                "total_results": len(snippets),
                "query_count": 1,
            }
            if parameters.get("fieldsOfStudy"):
                metrics["fields_of_study_ignored"] = True
            return ToolResponse(text=self._wrap_tool_output(blocks)), 0.0, metrics
        except Exception as e:
            return ToolResponse(text=f"Error performing snippet_search: {e}"), 0.0, {"status": "error", "error": str(e)}


class BrowseWebpageTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="browse_webpage",
                description="Open a specific URL and extract readable page text.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "url": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The webpage URL to browse.",
                        ),
                    },
                    required=["url"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = self._clean_text(parameters.get("url"))
        if not url:
            return ToolResponse(text="Error: url is required."), 0.0, {"status": "error"}

        backend = str(self.config.get("backend", "jina")).lower()

        def _browse():
            if backend == "serper":
                response = requests.post(
                    "https://scrape.serper.dev",
                    headers={
                        "X-API-KEY": _require_env_var("SERPER_API_KEY"),
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(
                        {
                            "url": url,
                            "includeMarkdown": bool(self.config.get("include_markdown", True)),
                        }
                    ),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                return {
                    "url": url,
                    "title": payload.get("metadata", {}).get("title", ""),
                    "content": payload.get("markdown") or payload.get("text") or "",
                }

            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers={
                    "Authorization": f"Bearer {_require_env_var('JINA_API_KEY')}",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json().get("data", {})
            return {
                "url": payload.get("url", url),
                "title": payload.get("title", ""),
                "content": payload.get("content", ""),
            }

        try:
            payload = await asyncio.to_thread(_browse)
            result_id = self._next_result_id(instance_id, "webpage")
            lines = [
                f"Title: {self._clean_text(payload.get('title')) or 'No title available'}",
                f"URL: {self._clean_text(payload.get('url')) or url}",
                f"Snippet: {self._clean_text(payload.get('content')) or 'No content available'}",
            ]
            text = self._wrap_tool_output([f"<webpage id={result_id}>\n" + "\n".join(lines) + "\n</webpage>"])
            return ToolResponse(text=text), 0.0, {"status": "success", "query_count": 1, "backend": backend}
        except Exception as e:
            return ToolResponse(text=f"Error performing browse_webpage: {e}"), 0.0, {"status": "error", "error": str(e)}
