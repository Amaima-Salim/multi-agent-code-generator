import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="AI Dev Team v2")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_projects: dict[str, dict] = {}
_queues: dict[str, queue.Queue] = {}


class GenerateRequest(BaseModel):
    requirement: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if not req.requirement.strip():
        return {"error": "requirement cannot be empty"}

    project_id = str(uuid.uuid4())[:8]
    q: queue.Queue = queue.Queue()
    _queues[project_id] = q

    def run():
        from graph.pipeline import build_pipeline
        from graph.state import create_initial_state

        pipeline = build_pipeline()
        initial = create_initial_state(req.requirement.strip(), project_id)
        config = {"configurable": {"queue": q}}

        final_state: dict | None = None
        try:
            # stream_mode="values" yields full state snapshot after each node
            for snapshot in pipeline.stream(initial, config=config, stream_mode="values"):
                final_state = snapshot
        except Exception as exc:
            q.put({"type": "error", "message": str(exc)})
            return

        if final_state:
            _projects[project_id] = final_state

        q.put({
            "type": "done",
            "project_id": project_id,
            "status": (final_state or {}).get("status", "failed"),
            "output_path": (final_state or {}).get("output_path"),
        })

    threading.Thread(target=run, daemon=True).start()
    return {"project_id": project_id}


@app.get("/api/stream/{project_id}")
async def stream_events(project_id: str):
    if project_id not in _queues:
        return {"error": "unknown project"}

    q = _queues[project_id]

    async def generator():
        while True:
            try:
                event = q.get(timeout=120)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error", "timeout"):
                    _queues.pop(project_id, None)
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
                break

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/project/{project_id}")
async def get_project(project_id: str):
    state = _projects.get(project_id)
    if not state:
        return {"error": "project not found"}
    return {
        "project_id": project_id,
        "requirement": state.get("requirement"),
        "status": state.get("status"),
        "spec": state.get("spec"),
        "architecture": state.get("architecture"),
        "code": state.get("code"),
        "qa_report": state.get("qa_report"),
        "review": state.get("review"),
        "output_path": state.get("output_path"),
        "memory_hits": state.get("memory_hits", []),
        "search_queries": state.get("search_queries", []),
        "llm_log": state.get("llm_log", []),
        "logs": state.get("logs", []),
    }


@app.get("/api/memory/stats")
async def memory_stats():
    try:
        from memory.chroma import MemoryManager
        return MemoryManager().stats()
    except Exception as exc:
        return {"error": str(exc)}
