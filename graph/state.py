from typing import TypedDict, Optional, Annotated
import operator


class GraphState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────
    requirement: str
    project_id: str

    # ── Agent outputs (filled progressively) ────────────────
    spec: Optional[dict]
    architecture: Optional[dict]
    code: Optional[dict]
    qa_report: Optional[dict]
    review: Optional[dict]

    # ── Enrichment (accumulated across nodes) ───────────────
    # ChromaDB hits surfaced to QA agent
    memory_hits: Annotated[list[str], operator.add]
    # Tavily queries made by Developer agent
    search_queries: Annotated[list[str], operator.add]
    # Which LLM each agent used  {"PM Agent": "claude-opus-4-7", ...}
    llm_log: Annotated[list[str], operator.add]

    # ── Pipeline metadata ────────────────────────────────────
    output_path: Optional[str]
    status: str
    logs: Annotated[list[str], operator.add]


def create_initial_state(requirement: str, project_id: str) -> GraphState:
    return {
        "requirement": requirement,
        "project_id": project_id,
        "spec": None,
        "architecture": None,
        "code": None,
        "qa_report": None,
        "review": None,
        "memory_hits": [],
        "search_queries": [],
        "llm_log": [],
        "output_path": None,
        "status": "running",
        "logs": [],
    }
