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

    def _get_runway_url(self) -> str:
        return self._clean_text(
            self.config.get(
                "url",
                os.getenv(
                    "RUNWAY_WEB_SEARCH_API_URL",
                    "https://runway.devops.xiaohongshu.com/openai/zhipu/paas/v4/web_search",
                ),
            )
        )

    def _get_runway_api_key(self) -> str:
        api_key_env = self._clean_text(self.config.get("api_key_env")) or "RUNWAY_WEB_SEARCH_API_KEY"
        return _require_env_var(api_key_env)

    @staticmethod
    def _get_nested_value(payload: Any, path: tuple[str, ...]) -> Any:
        current = payload
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    @classmethod
    def _first_text_value(cls, payload: Any, paths: list[tuple[str, ...]]) -> str:
        for path in paths:
            value = cls._get_nested_value(payload, path)
            text = cls._clean_text(value)
            if text:
                return text
        return ""

    @classmethod
    def _score_result_record(cls, record: dict[str, Any]) -> int:
        score = 0
        if cls._clean_text(record.get("title") or record.get("name")):
            score += 3
        if cls._clean_text(record.get("url") or record.get("link") or record.get("href")):
            score += 3
        if cls._clean_text(
            record.get("snippet")
            or record.get("description")
            or record.get("summary")
            or record.get("text")
            or record.get("content")
            or record.get("body")
        ):
            score += 2
        if cls._clean_text(
            record.get("date")
            or record.get("publishedDate")
            or record.get("time")
            or record.get("publish_time")
        ):
            score += 1
        return score

    @classmethod
    def _preferred_candidate_bonus(cls, path: tuple[str, ...]) -> int:
        bonuses = {
            "organic": 6,
            "results": 5,
            "items": 4,
            "documents": 3,
            "docs": 3,
            "pages": 2,
            "value": 1,
            "data": 0,
        }
        return sum(bonuses.get(segment, 0) for segment in path)

    @classmethod
    def _collect_dict_list_candidates(
        cls, payload: Any, path: tuple[str, ...] = ()
    ) -> list[tuple[tuple[str, ...], list[dict[str, Any]]]]:
        candidates: list[tuple[tuple[str, ...], list[dict[str, Any]]]] = []
        if isinstance(payload, list):
            dict_items = [item for item in payload if isinstance(item, dict)]
            if dict_items:
                candidates.append((path, dict_items))
            for index, item in enumerate(payload):
                candidates.extend(cls._collect_dict_list_candidates(item, path + (str(index),)))
            return candidates

        if isinstance(payload, dict):
            preferred_keys = ["organic", "results", "items", "data", "documents", "docs", "value", "pages"]
            iter_keys = preferred_keys + [key for key in payload.keys() if key not in preferred_keys]
            for key in iter_keys:
                if key in payload:
                    candidates.extend(cls._collect_dict_list_candidates(payload[key], path + (key,)))
        return candidates

    @classmethod
    def _find_list_of_dicts(cls, payload: Any) -> list[dict[str, Any]]:
        candidates = cls._collect_dict_list_candidates(payload)
        best_records: list[dict[str, Any]] = []
        best_score: tuple[float, int, int] | None = None

        for path, records in candidates:
            record_scores = [cls._score_result_record(record) for record in records]
            total_score = sum(record_scores)
            max_score = max(record_scores, default=0)
            bonus = cls._preferred_candidate_bonus(path)
            candidate_score = (max_score, total_score + bonus, len(records))
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_records = records

        if best_score is None or best_score[0] <= 0:
            return []
        return best_records

    @classmethod
    def _extract_result_records(cls, payload: Any) -> list[dict[str, Any]]:
        records = cls._find_list_of_dicts(payload)
        if records:
            return records
        if isinstance(payload, dict):
            likely_record = {
                "title": cls._first_text_value(payload, [("title",), ("name",)]),
                "url": cls._first_text_value(payload, [("url",), ("link",), ("href",)]),
                "snippet": cls._first_text_value(
                    payload,
                    [
                        ("snippet",),
                        ("description",),
                        ("summary",),
                        ("text",),
                        ("content",),
                        ("body",),
                    ],
                ),
            }
            if any(likely_record.values()):
                return [likely_record]
        return []

    @classmethod
    def _extract_webpage_payload(cls, payload: Any, fallback_url: str) -> dict[str, str]:
        title = cls._first_text_value(
            payload,
            [
                ("title",),
                ("data", "title"),
                ("metadata", "title"),
                ("result", "title"),
            ],
        )
        page_url = cls._first_text_value(
            payload,
            [
                ("url",),
                ("data", "url"),
                ("metadata", "url"),
                ("result", "url"),
            ],
        ) or fallback_url
        content = cls._first_text_value(
            payload,
            [
                ("content",),
                ("text",),
                ("markdown",),
                ("body",),
                ("result",),
                ("data", "content"),
                ("data", "text"),
                ("data", "markdown"),
                ("data", "body"),
                ("data", "result"),
            ],
        )

        if not content:
            records = cls._extract_result_records(payload)
            if records:
                joined = []
                for record in records[:3]:
                    lines = []
                    title_text = cls._clean_text(record.get("title") or record.get("name"))
                    url_text = cls._clean_text(record.get("url") or record.get("link") or record.get("href"))
                    content_text = cls._clean_text(
                        record.get("content")
                        or record.get("text")
                        or record.get("snippet")
                        or record.get("description")
                        or record.get("summary")
                    )
                    if title_text:
                        lines.append(f"Title: {title_text}")
                    if url_text:
                        lines.append(f"URL: {url_text}")
                    if content_text:
                        lines.append(f"Content: {content_text}")
                    if lines:
                        joined.append("\n".join(lines))
                content = "\n\n".join(joined)

        if not content:
            content = cls._clean_text(payload)

        return {
            "url": page_url,
            "title": title,
            "content": content,
        }

    @classmethod
    def _extract_runway_web_reader_payload(cls, payload: Any, fallback_url: str) -> dict[str, str]:
        title = cls._first_text_value(
            payload,
            [
                ("title",),
                ("data", "title"),
                ("metadata", "title"),
            ],
        )
        page_url = cls._first_text_value(
            payload,
            [
                ("url",),
                ("data", "url"),
                ("metadata", "url"),
            ],
        ) or fallback_url
        content = cls._first_text_value(
            payload,
            [
                ("content",),
                ("text",),
                ("markdown",),
                ("result",),
                ("data", "content"),
                ("data", "text"),
                ("data", "markdown"),
                ("data", "result"),
            ],
        )

        if not content:
            content = cls._clean_text(payload)

        return {
            "url": page_url,
            "title": title,
            "content": content,
        }

    def _runway_request(self, payload: dict[str, Any]) -> Any:
        response = requests.post(
            self._get_runway_url(),
            headers={
                "api-key": self._get_runway_api_key(),
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _get_runway_search_engine(self, default: str) -> str:
        return self._clean_text(self.config.get("search_engine")) or default


class GoogleSearchTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        backend = self._clean_text(self.config.get("backend")).lower() or "serper"
        properties = {
            "query": OpenAIFunctionPropertySchema(
                type="string",
                description="The search query.",
            ),
        }
        if backend != "runway":
            properties.update(
                {
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
                }
            )

        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="google_search",
                description="General web search for relevant webpages and snippets.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties=properties,
                    required=["query"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = self._clean_text(parameters.get("query"))
        if not query:
            return ToolResponse(text="Error: query is required."), 0.0, {"status": "error"}

        backend = self._clean_text(self.config.get("backend")).lower() or "serper"

        def _search():
            if backend == "runway":
                search_engine = self._get_runway_search_engine("search_prime")
                if search_engine != "search_prime":
                    raise ValueError(
                        f"google_search with runway backend must use search_engine='search_prime', got '{search_engine}'"
                    )
                payload = self._runway_request(
                    {
                        "search_engine": search_engine,
                        "search_query": query,
                        "query_rewrite": str(self.config.get("query_rewrite", "false")).lower(),
                    }
                )
                return payload

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
            organic_results = self._extract_result_records(payload)
            blocks = []
            for result in organic_results:
                result_id = self._next_result_id(instance_id, "snippet")
                lines = [
                    f"Title: {self._clean_text(result.get('title') or result.get('name'))}",
                    f"URL: {self._clean_text(result.get('link') or result.get('url') or result.get('href'))}",
                    (
                        "Search Snippet: "
                        f"{self._clean_text(result.get('snippet') or result.get('description') or result.get('summary') or result.get('text') or result.get('content'))}"
                    ),
                ]
                result_date = self._clean_text(
                    result.get("date")
                    or result.get("publishedDate")
                    or result.get("time")
                    or result.get("publish_time")
                )
                if result_date:
                    lines.append(f"Date: {result_date}")
                blocks.append(f"<snippet id={result_id}>\n" + "\n".join(lines) + "\n</snippet>")
            if not blocks:
                empty_id = self._next_result_id(instance_id, "snippet")
                blocks.append(f"<snippet id={empty_id}>\nNo search results found.\n</snippet>")
            return (
                ToolResponse(text=self._wrap_tool_output(blocks)),
                0.0,
                {"status": "success", "total_results": len(organic_results), "query_count": 1, "backend": backend},
            )
        except Exception as e:
            return ToolResponse(text=f"Error performing google_search: {e}"), 0.0, {"status": "error", "error": str(e)}


class SnippetSearchTool(_DrTuluBaseTool):
    def _build_fallback_response(
        self,
        instance_id: str,
        query: str,
        reason: str,
        parameters: dict[str, Any],
    ) -> tuple[ToolResponse, float, dict]:
        result_id = self._next_result_id(instance_id, "snippet")
        fallback_message = self._clean_text(self.config.get("fallback_message")) or (
            "Semantic Scholar API is unavailable in this environment. "
            "This placeholder snippet is only for pipeline debugging and should not be treated as factual evidence. "
            "Prefer google_search and browse_webpage for actual retrieval."
        )
        lines = [
            "Title: Semantic Scholar placeholder (debug mode)",
            "Paper ID: debug-placeholder",
            f"Snippet: {fallback_message} Query: {query}",
        ]
        if parameters.get("year"):
            lines.append(f"Requested Year Filter: {self._clean_text(parameters.get('year'))}")
        if parameters.get("fieldsOfStudy"):
            lines.append(f"Requested Fields Of Study: {self._clean_text(parameters.get('fieldsOfStudy'))}")
        if parameters.get("venue"):
            lines.append(f"Requested Venue: {self._clean_text(parameters.get('venue'))}")

        metrics = {
            "status": "success",
            "total_results": 1,
            "query_count": 1,
            "fallback_used": True,
            "fallback_reason": reason,
        }
        if parameters.get("fieldsOfStudy"):
            metrics["fields_of_study_ignored"] = True

        text = self._wrap_tool_output([f"<snippet id={result_id}>\n" + "\n".join(lines) + "\n</snippet>"])
        return ToolResponse(text=text), 0.0, metrics

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

        api_key = os.getenv("S2_API_KEY")
        fallback_on_missing_api_key = bool(self.config.get("fallback_on_missing_api_key", True))
        fallback_on_request_error = bool(self.config.get("fallback_on_request_error", True))

        if not api_key and fallback_on_missing_api_key:
            return self._build_fallback_response(
                instance_id=instance_id,
                query=query,
                reason="missing_s2_api_key",
                parameters=parameters,
            )

        def _search():
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/snippet/search",
                params={
                    "query": query,
                    "limit": int(parameters.get("limit", self.config.get("limit", 5))),
                    **({"year": parameters["year"]} if parameters.get("year") else {}),
                    **({"venue": parameters["venue"]} if parameters.get("venue") else {}),
                },
                headers={"x-api-key": api_key} if api_key else None,
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
            if fallback_on_request_error:
                return self._build_fallback_response(
                    instance_id=instance_id,
                    query=query,
                    reason=f"request_error:{type(e).__name__}",
                    parameters=parameters,
                )
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

        backend = self._clean_text(self.config.get("backend")).lower() or "jina"

        def _browse():
            if backend == "runway":
                search_engine = self._get_runway_search_engine("web-reader")
                if search_engine != "web-reader":
                    raise ValueError(
                        f"browse_webpage with runway backend must use search_engine='web-reader', got '{search_engine}'"
                    )
                payload = self._runway_request(
                    {
                        "search_engine": search_engine,
                        "url": url,
                    }
                )
                return self._extract_runway_web_reader_payload(payload, fallback_url=url)

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


class DownloadFileTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="download_file",
                description="Compatibility fallback for hallucinated file download calls.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "url": OpenAIFunctionPropertySchema(
                            type="string",
                            description="The file or webpage URL to access.",
                        ),
                    },
                    required=["url"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = self._clean_text(parameters.get("url"))
        result_id = self._next_result_id(instance_id, "webpage")
        lines = [
            "Title: download_file is unsupported in this training setup",
            f"URL: {url or 'No URL provided'}",
            (
                "Snippet: Please use browse_webpage directly with the target URL instead of download_file. "
                "This fallback exists only to keep the rollout running when the model hallucinates the wrong tool name."
            ),
        ]
        text = self._wrap_tool_output([f"<webpage id={result_id}>\n" + "\n".join(lines) + "\n</webpage>"])
        metrics = {
            "status": "success",
            "query_count": 1,
            "fallback_used": True,
            "fallback_reason": "hallucinated_download_file_tool",
        }
        return ToolResponse(text=text), 0.0, metrics
