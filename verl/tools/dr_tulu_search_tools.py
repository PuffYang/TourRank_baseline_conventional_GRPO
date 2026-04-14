import asyncio
import fcntl
import json
import logging
import os
import tempfile
import time
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests

from .base_tool import BaseTool
from .schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema
from .schemas import OpenAIFunctionToolSchema, ToolResponse

DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
logger = logging.getLogger(__name__)


class _RetriableRequestError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: Optional[float] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


class _FileBackedRateLimiter:
    def __init__(self, lock_path: str, min_interval_seconds: float):
        self.lock_path = lock_path
        self.min_interval_seconds = max(float(min_interval_seconds), 0.0)

    def wait(self) -> float:
        if self.min_interval_seconds <= 0:
            return 0.0

        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        with open(self.lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read().strip()
            now = time.time()
            last_ts = 0.0
            if raw:
                try:
                    last_ts = float(raw)
                except ValueError:
                    last_ts = 0.0

            wait_seconds = max(0.0, last_ts + self.min_interval_seconds - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            next_ts = time.time()
            handle.seek(0)
            handle.truncate()
            handle.write(f"{next_ts:.6f}")
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return wait_seconds

    def run_serialized(self, fn):
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        with open(self.lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read().strip()
            now = time.time()
            last_ts = 0.0
            if raw:
                try:
                    last_ts = float(raw)
                except ValueError:
                    last_ts = 0.0

            wait_seconds = max(0.0, last_ts + self.min_interval_seconds - now)
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            try:
                result = fn(wait_seconds)
            finally:
                next_ts = time.time()
                handle.seek(0)
                handle.truncate()
                handle.write(f"{next_ts:.6f}")
                handle.flush()
                os.fsync(handle.fileno())
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        return result


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

    @classmethod
    def _normalize_url(cls, url: Any) -> str:
        normalized = cls._clean_text(url)
        if not normalized:
            return ""
        if "://" not in normalized:
            normalized = f"https://{normalized.lstrip('/')}"
        parsed = urlparse(normalized)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {normalized}")
        return normalized


class GoogleSearchTool(_DrTuluBaseTool):
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        properties = {
            "query": OpenAIFunctionPropertySchema(
                type="string",
                description="The search query.",
            ),
        }

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

        def _search():
            search_engine = self._get_runway_search_engine("search_prime")
            if search_engine != "search_prime":
                raise ValueError(
                    f"google_search must use search_engine='search_prime', got '{search_engine}'"
                )
            payload = self._runway_request(
                {
                    "search_engine": search_engine,
                    "search_query": query,
                    "query_rewrite": str(self.config.get("query_rewrite", "false")).lower(),
                }
            )
            return payload

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
                {"status": "success", "total_results": len(organic_results), "query_count": 1, "backend": "runway"},
            )
        except Exception as e:
            return ToolResponse(text=f"Error performing google_search: {e}"), 0.0, {"status": "error", "error": str(e)}


class SnippetSearchTool(_DrTuluBaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema)
        min_interval_seconds = float(
            config.get("min_interval_seconds", os.getenv("S2_MIN_INTERVAL_SECONDS", "1.1"))
        )
        rate_limit_file = self._clean_text(
            config.get(
                "rate_limit_file",
                os.getenv(
                    "S2_RATE_LIMIT_FILE",
                    os.path.join(tempfile.gettempdir(), "verl_snippet_search_rate_limit.txt"),
                ),
            )
        )
        self._rate_limiter = _FileBackedRateLimiter(rate_limit_file, min_interval_seconds)
        self.max_retries = max(1, int(config.get("max_retries", os.getenv("S2_MAX_RETRIES", "3"))))
        self.retry_backoff_seconds = max(
            0.0, float(config.get("retry_backoff_seconds", os.getenv("S2_RETRY_BACKOFF_SECONDS", "1.0")))
        )
        self.serialize_requests = _as_bool(config.get("serialize_requests", True), True)
        self.retriable_status_codes = {429, 500, 502, 503, 504}

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
        fallback_on_missing_api_key = _as_bool(self.config.get("fallback_on_missing_api_key", True), True)
        fallback_on_request_error = _as_bool(self.config.get("fallback_on_request_error", True), True)

        if not api_key:
            if fallback_on_missing_api_key:
                logger.warning(
                    "snippet_search fallback triggered: reason=missing_s2_api_key query=%r parameters=%s",
                    query,
                    {
                        "limit": parameters.get("limit"),
                        "year": parameters.get("year"),
                        "fieldsOfStudy": parameters.get("fieldsOfStudy"),
                        "venue": parameters.get("venue"),
                    },
                )
                return self._build_fallback_response(
                    instance_id=instance_id,
                    query=query,
                    reason="missing_s2_api_key",
                    parameters=parameters,
                )
            error_message = (
                "Error performing snippet_search: S2_API_KEY environment variable is not set. "
                "The verl tool implementation does not auto-load .env files; please export S2_API_KEY "
                "in the training environment before launching the job."
            )
            return ToolResponse(text=error_message), 0.0, {"status": "error", "error": error_message}

        def _search():
            params = {
                "query": query,
                "limit": int(parameters.get("limit", self.config.get("limit", 5))),
                **({"year": parameters["year"]} if parameters.get("year") else {}),
                **({"venue": parameters["venue"]} if parameters.get("venue") else {}),
            }

            def _attempt_request(waited: float):
                if waited > 0:
                    logger.info(
                        "snippet_search rate limiter slept %.2fs before query=%r",
                        waited,
                        query,
                    )

                last_error: Optional[Exception] = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = requests.get(
                            "https://api.semanticscholar.org/graph/v1/snippet/search",
                            params=params,
                            headers={"x-api-key": api_key},
                            timeout=self.timeout,
                        )
                        if response.status_code in self.retriable_status_codes:
                            retry_after = response.headers.get("Retry-After")
                            retry_after_seconds = None
                            if retry_after:
                                try:
                                    retry_after_seconds = float(retry_after)
                                except ValueError:
                                    retry_after_seconds = None
                            raise _RetriableRequestError(
                                (
                                    "Semantic Scholar snippet_search returned retryable status "
                                    f"{response.status_code}: {response.text[:300]}"
                                ),
                                retry_after_seconds=retry_after_seconds,
                            )
                        response.raise_for_status()
                        return response.json()
                    except (
                        requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError,
                        requests.exceptions.ReadTimeout,
                        _RetriableRequestError,
                    ) as e:
                        last_error = e
                        if attempt >= self.max_retries:
                            break
                        sleep_seconds = getattr(e, "retry_after_seconds", None)
                        if sleep_seconds is None:
                            sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                        logger.warning(
                            "snippet_search retrying query=%r attempt=%d/%d sleep=%.2fs error_type=%s error=%s",
                            query,
                            attempt,
                            self.max_retries,
                            sleep_seconds,
                            type(e).__name__,
                            str(e),
                        )
                        time.sleep(sleep_seconds)

                if last_error is None:
                    raise RuntimeError("snippet_search failed without an explicit exception")
                raise last_error

            if self.serialize_requests:
                return self._rate_limiter.run_serialized(_attempt_request)

            waited = self._rate_limiter.wait()
            return _attempt_request(waited)

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
                logger.warning(
                    "snippet_search fallback triggered: reason=request_error query=%r error_type=%s error=%s parameters=%s",
                    query,
                    type(e).__name__,
                    str(e),
                    {
                        "limit": parameters.get("limit"),
                        "year": parameters.get("year"),
                        "fieldsOfStudy": parameters.get("fieldsOfStudy"),
                        "venue": parameters.get("venue"),
                    },
                )
                return self._build_fallback_response(
                    instance_id=instance_id,
                    query=query,
                    reason=f"request_error:{type(e).__name__}",
                    parameters=parameters,
                )
            error_message = (
                "Error performing snippet_search: "
                f"{type(e).__name__}: {e}. "
                "This request was not replaced with a placeholder because fallback_on_request_error is disabled."
            )
            return ToolResponse(text=error_message), 0.0, {"status": "error", "error": error_message}


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

        normalized_url = self._normalize_url(url)

        def _browse() -> dict[str, str]:
            response = requests.get(
                f"https://r.jina.ai/{normalized_url}",
                timeout=self.timeout,
            )
            response.raise_for_status()

            content_type = self._clean_text(response.headers.get("Content-Type")).lower()
            if "json" in content_type:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    extracted = self._extract_webpage_payload(payload.get("data", payload), fallback_url=normalized_url)
                    if self._clean_text(extracted.get("content")):
                        return extracted

            text_content = self._clean_text(response.text)
            if not text_content:
                raise ValueError("Jina Reader response did not include readable page content")
            return {
                "url": normalized_url,
                "title": "",
                "content": text_content,
            }

        try:
            payload = await asyncio.to_thread(_browse)
            result_id = self._next_result_id(instance_id, "webpage")
            lines = [
                f"Title: {self._clean_text(payload.get('title')) or 'No title available'}",
                f"URL: {self._clean_text(payload.get('url')) or normalized_url}",
                f"Content: {self._clean_text(payload.get('content')) or 'No content available'}",
            ]
            text = self._wrap_tool_output([f"<webpage id={result_id}>\n" + "\n".join(lines) + "\n</webpage>"])
            return ToolResponse(text=text), 0.0, {
                "status": "success",
                "query_count": 1,
                "backend": "jina",
            }
        except Exception as e:
            return ToolResponse(text=f"Error performing browse_webpage: {e}"), 0.0, {"status": "error", "error": str(e)}


