"""
LangGraph pipeline — the heart of ai-dev-team-v2.

Each agent is a node in a StateGraph. Nodes receive the full GraphState and
return a partial update dict. LangGraph merges updates automatically, including
appending to Annotated[list, operator.add] fields (logs, memory_hits, etc.).

Progress is streamed to the caller via a queue passed through LangGraph's
configurable mechanism:
    config = {"configurable": {"queue": q}}
    pipeline.stream(state, config=config, stream_mode="values")
"""
from __future__ import annotations

import queue as _queue
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from graph.state import GraphState


# ── Node helpers ─────────────────────────────────────────────────────────────

def _q(config: RunnableConfig) -> _queue.Queue | None:
    return config.get("configurable", {}).get("queue")


def _push(config: RunnableConfig, payload: dict) -> None:
    q = _q(config)
    if q is not None:
        q.put(payload)


# ── Node: PM Agent ───────────────────────────────────────────────────────────

def pm_node(state: GraphState, config: RunnableConfig) -> dict:
    _push(config, {"type": "node_start", "node": "PM Agent"})

    from agents.pm import PMAgent
    agent = PMAgent()
    spec = agent.run(state["requirement"])

    _push(config, {"type": "node_complete", "node": "PM Agent",
                   "detail": spec.title})
    return {
        "spec": spec.model_dump(),
        "llm_log": [f"PM Agent → {agent.llm_label}"],
        "logs": [f"📋 PM Agent: '{spec.title}' · {len(spec.features)} features"],
    }


# ── Node: Architect Agent ────────────────────────────────────────────────────

def architect_node(state: GraphState, config: RunnableConfig) -> dict:
    _push(config, {"type": "node_start", "node": "Architect Agent"})

    from agents.architect import ArchitectAgent
    from memory.chroma import MemoryManager
    from models.schemas import ProductSpec

    memory = MemoryManager()
    past_decisions = memory.query_arch_decisions(state["requirement"], n=3)

    agent = ArchitectAgent()
    spec = ProductSpec(**state["spec"])
    arch = agent.run(spec, past_decisions=past_decisions or None)

    # Store new decisions for future runs
    memory.store_arch_decisions(
        state["project_id"],
        [d.model_dump() for d in arch.key_decisions],
        arch.tech_stack,
    )

    _push(config, {"type": "node_complete", "node": "Architect Agent",
                   "detail": f"{len(arch.tech_stack)} technologies"})
    return {
        "architecture": arch.model_dump(),
        "llm_log": [f"Architect Agent → {agent.llm_label}"],
        "logs": [f"🏗️  Architect: {len(arch.tech_stack)} tech · {len(arch.file_structure)} files"],
    }


# ── Node: Developer Agent ────────────────────────────────────────────────────

def developer_node(state: GraphState, config: RunnableConfig) -> dict:
    _push(config, {"type": "node_start", "node": "Developer Agent"})

    from agents.developer import DeveloperAgent
    from tools.search import SearchTool
    from models.schemas import ProductSpec, Architecture

    search = SearchTool()
    agent = DeveloperAgent(search_tool=search)
    code = agent.run(
        ProductSpec(**state["spec"]),
        Architecture(**state["architecture"]),
    )

    _push(config, {"type": "node_complete", "node": "Developer Agent",
                   "detail": f"{len(code.files)} files"})
    return {
        "code": code.model_dump(),
        "search_queries": agent.searches_made,
        "llm_log": [f"Developer Agent → {agent.llm_label}"],
        "logs": [
            f"💻 Developer: {len(code.files)} files written"
            + (f" · {len(agent.searches_made)} Tavily searches" if agent.searches_made else "")
        ],
    }


# ── Node: QA Agent ───────────────────────────────────────────────────────────

def qa_node(state: GraphState, config: RunnableConfig) -> dict:
    _push(config, {"type": "node_start", "node": "QA Agent"})

    from agents.qa import QAAgent
    from memory.chroma import MemoryManager
    from models.schemas import ProductSpec, Architecture, CodeOutput

    arch_data = state.get("architecture") or {}
    tech_stack = arch_data.get("tech_stack", [])

    # Query ChromaDB for relevant past issues
    memory = MemoryManager()
    past_issues = memory.query_similar_issues(state["requirement"], tech_stack, n=5)

    agent = QAAgent()
    qa = agent.run(
        ProductSpec(**state["spec"]),
        Architecture(**state["architecture"]),
        CodeOutput(**state["code"]),
        memory_context=past_issues or None,
    )

    # Store new issues for future runs
    memory.store_issues(
        state["project_id"],
        [i.model_dump() for i in qa.issues],
        tech_stack,
    )

    _push(config, {"type": "node_complete", "node": "QA Agent",
                   "detail": f"{qa.overall_score}/10"})
    return {
        "qa_report": qa.model_dump(),
        "memory_hits": past_issues,
        "llm_log": [f"QA Agent → {agent.llm_label}"],
        "logs": [
            f"🧪 QA: {qa.overall_score}/10 · {len(qa.issues)} issues"
            + (f" · {len(past_issues)} memory hits" if past_issues else "")
        ],
    }


# ── Node: Reviewer Agent ─────────────────────────────────────────────────────

def reviewer_node(state: GraphState, config: RunnableConfig) -> dict:
    _push(config, {"type": "node_start", "node": "Reviewer Agent"})

    from agents.reviewer import ReviewerAgent
    from models.schemas import ProductSpec, CodeOutput, QAReport

    agent = ReviewerAgent()
    review = agent.run(
        ProductSpec(**state["spec"]),
        CodeOutput(**state["code"]),
        QAReport(**state["qa_report"]),
    )

    approved = "✅ Approved" if review.approved else "❌ Changes Requested"
    _push(config, {"type": "node_complete", "node": "Reviewer Agent",
                   "detail": approved})
    return {
        "review": review.model_dump(),
        "llm_log": [f"Reviewer Agent → {agent.llm_label}"],
        "logs": [f"🔍 Reviewer: {approved} · Maintainability {review.maintainability_score}/10"],
    }


# ── Node: Save output ────────────────────────────────────────────────────────

def save_node(state: GraphState, config: RunnableConfig) -> dict:
    _push(config, {"type": "node_start", "node": "Save Output"})

    from tools.file_manager import FileManager

    path = FileManager().save_project(state)

    _push(config, {"type": "node_complete", "node": "Save Output",
                   "detail": str(path)})
    return {
        "output_path": str(path),
        "status": "completed",
        "logs": [f"💾 Saved → {path}"],
    }


# ── Build the graph ───────────────────────────────────────────────────────────

def build_pipeline():
    g = StateGraph(GraphState)

    g.add_node("pm",        pm_node)
    g.add_node("architect", architect_node)
    g.add_node("developer", developer_node)
    g.add_node("qa",        qa_node)
    g.add_node("reviewer",  reviewer_node)
    g.add_node("save",      save_node)

    g.add_edge(START,       "pm")
    g.add_edge("pm",        "architect")
    g.add_edge("architect", "developer")
    g.add_edge("developer", "qa")
    g.add_edge("qa",        "reviewer")
    g.add_edge("reviewer",  "save")
    g.add_edge("save",      END)

    return g.compile()
