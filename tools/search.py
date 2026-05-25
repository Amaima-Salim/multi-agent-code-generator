"""
Tavily search tool for the Developer agent.

If TAVILY_API_KEY is not set, all methods return empty results gracefully
so the pipeline continues without web search.
"""
from __future__ import annotations

from rich.console import Console

console = Console()


class SearchTool:
    def __init__(self) -> None:
        from config import settings

        self._enabled = settings.has_tavily
        self._client = None

        if self._enabled:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=settings.tavily_api_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        if not self._enabled or not self._client:
            return []
        try:
            resp = self._client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
            )
            return resp.get("results", [])
        except Exception as exc:
            console.print(f"  [yellow]⚠ Tavily search failed: {exc}[/yellow]")
            return []

    def search_for_docs(self, technology: str, topic: str) -> str:
        """
        Return a compact string of relevant search snippets for a given
        technology + topic (e.g. 'FastAPI' + 'SQLAlchemy async setup').
        Returns an empty string when Tavily is not available.
        """
        if not self._enabled:
            return ""

        query = f"{technology} {topic} python code example documentation"
        results = self.search(query, max_results=3)
        if not results:
            return ""

        parts = []
        for r in results:
            snippet = r.get("content", "")[:400]
            url = r.get("url", "")
            if snippet:
                parts.append(f"[{url}]\n{snippet}")

        return "\n\n".join(parts)
