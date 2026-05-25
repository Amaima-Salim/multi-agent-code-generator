"""
ChromaDB-backed persistent memory for the AI Dev Team pipeline.

Two collections:
  qa_issues      — bugs / issues found across past projects
  arch_decisions — architecture decisions and their rationales

The QA agent queries qa_issues before reviewing code (so it catches
recurring patterns), then stores new findings after the review.

The Architect agent queries arch_decisions to inform its choices.
"""
from __future__ import annotations

import chromadb
from rich.console import Console

console = Console()


class MemoryManager:
    def __init__(self) -> None:
        from config import MEMORY_DIR

        self._client = chromadb.PersistentClient(path=str(MEMORY_DIR))

        self._qa = self._client.get_or_create_collection(
            name="qa_issues",
            metadata={"hnsw:space": "cosine"},
        )
        self._arch = self._client.get_or_create_collection(
            name="arch_decisions",
            metadata={"hnsw:space": "cosine"},
        )

    # ── QA issues ────────────────────────────────────────────────────────────

    def query_similar_issues(
        self, requirement: str, tech_stack: list[str], n: int = 5
    ) -> list[str]:
        """Return past QA issues relevant to this project's tech stack."""
        count = self._qa.count()
        if count == 0:
            return []
        query = f"{requirement} {' '.join(tech_stack)}"
        results = self._qa.query(
            query_texts=[query],
            n_results=min(n, count),
        )
        return results["documents"][0] if results["documents"] else []

    def store_issues(
        self, project_id: str, issues: list[dict], tech_stack: list[str]
    ) -> None:
        """Persist QA issues so future runs can learn from them."""
        tech_str = ",".join(tech_stack[:6])
        for i, issue in enumerate(issues[:25]):
            doc = (
                f"[{issue.get('severity', 'info').upper()}] "
                f"{issue.get('file', '?')}: {issue.get('description', '')} "
                f"→ {issue.get('suggestion', '')}"
            )
            uid = f"{project_id}_issue_{i}"
            try:
                self._qa.add(
                    documents=[doc],
                    metadatas=[{"project_id": project_id, "tech": tech_str,
                                "severity": issue.get("severity", "info")}],
                    ids=[uid],
                )
            except Exception:
                pass  # duplicate ID — skip silently

        console.print(
            f"  [dim]🧠 Memory:[/dim] stored {min(len(issues), 25)} QA issues "
            f"(total in DB: {self._qa.count()})"
        )

    # ── Architecture decisions ────────────────────────────────────────────────

    def query_arch_decisions(self, requirement: str, n: int = 3) -> list[str]:
        """Return past architecture decisions relevant to this requirement."""
        count = self._arch.count()
        if count == 0:
            return []
        results = self._arch.query(
            query_texts=[requirement],
            n_results=min(n, count),
        )
        return results["documents"][0] if results["documents"] else []

    def store_arch_decisions(
        self, project_id: str, decisions: list[dict], tech_stack: list[str]
    ) -> None:
        tech_str = ",".join(tech_stack[:6])
        for i, d in enumerate(decisions[:10]):
            doc = f"{d.get('decision', '')}: {d.get('rationale', '')}"
            uid = f"{project_id}_arch_{i}"
            try:
                self._arch.add(
                    documents=[doc],
                    metadatas=[{"project_id": project_id, "tech": tech_str}],
                    ids=[uid],
                )
            except Exception:
                pass

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "qa_issues": self._qa.count(),
            "arch_decisions": self._arch.count(),
        }
