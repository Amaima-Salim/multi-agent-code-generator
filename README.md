# multi-agent-code-generator

A multi-agent AI code generation system built on a production-grade stack. Given a natural language requirement, five specialized agents collaborate in a typed pipeline to produce a complete, reviewed codebase.

## Architecture

| Layer | Technology | Purpose |
|-------|------------|---------|
| Orchestration | LangGraph | Typed state machine managing agent graph execution |
| Memory | ChromaDB | Persistent vector store for cross-run issue learning |
| Web Search | Tavily | Live documentation retrieval during code generation |
| LLMs | Claude + GPT-4o | Per-agent model selection based on task type |
| API | FastAPI + SSE | Real-time pipeline streaming to the browser |

## Pipeline

```
START
  |
PM Agent        (Claude)    -- requirement -> ProductSpec
  |
Architect       (Claude)    -- spec -> Architecture  [reads ChromaDB arch decisions]
  |
Developer       (GPT-4o)    -- architecture -> CodeOutput  [Tavily search per file]
  |
QA Agent        (Claude)    -- code -> QAReport  [reads + writes ChromaDB issues]
  |
Reviewer        (Claude)    -- code + QA -> ReviewReport
  |
Save Output                 -- writes project to output/
  |
END
```

If `OPENAI_API_KEY` is not set, the Developer agent falls back to Claude.

## Key Features

**LangGraph state machine** -- The pipeline is implemented as a `StateGraph` with a typed `GraphState` (TypedDict). Each agent returns a partial state update; LangGraph handles merging, including list accumulation (`logs`, `memory_hits`, `search_queries`) via `Annotated[list, operator.add]`.

**Persistent ChromaDB memory** -- After each run, the QA agent stores discovered issues in a local vector database. On subsequent runs, it queries for similar past issues before reviewing code and injects them into the prompt. The system improves with use.

**Tavily web search** -- While writing each file, the Developer agent searches for relevant documentation and examples in real time, producing more accurate and up-to-date code.

**Multi-LLM routing** -- Each agent uses the model best suited to its task. The Developer agent uses GPT-4o; all reasoning and review agents use Claude.

## Setup

### Prerequisites

- Python 3.11 or higher
- Anthropic API key (required)
- OpenAI API key (optional, enables GPT-4o for Developer agent)
- Tavily API key (optional, enables web search; free tier available)

### Installation

```bash
cd ai-dev-team-v2

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Note: ChromaDB downloads a small embedding model (~50 MB) on first run.

### Configuration

```bash
cp .env.example .env
# Set the following in .env:
#   ANTHROPIC_API_KEY=sk-ant-...   (required)
#   OPENAI_API_KEY=sk-...          (optional)
#   TAVILY_API_KEY=tvly-...        (optional)
```

## Usage

### CLI

```bash
# Interactive mode
python main.py generate

# Direct prompt
python main.py generate "Build a URL shortener with FastAPI and SQLite"

# Inspect memory stats
python main.py memory
```

### Web UI

```bash
python main.py web
# Open http://127.0.0.1:8000
```

The web UI displays live pipeline progress, ChromaDB memory hits, Tavily search queries, generated files with syntax highlighting, and the QA and review reports.

## Project Structure

```
ai-dev-team-v2/
├── main.py              # CLI entry point (Typer) -- generate / web / memory commands
├── config.py            # Settings (Pydantic) -- detects optional API keys
│
├── graph/
│   ├── state.py         # GraphState TypedDict -- LangGraph pipeline state
│   └── pipeline.py      # StateGraph -- 6 nodes, queue-based SSE streaming
│
├── agents/
│   ├── base.py          # BaseAgent -- routes to Anthropic or OpenAI SDK
│   ├── pm.py            # PM Agent (Claude)
│   ├── architect.py     # Architect Agent (Claude) -- reads ChromaDB arch decisions
│   ├── developer.py     # Developer Agent (GPT-4o) -- Tavily search per file
│   ├── qa.py            # QA Agent (Claude) -- ChromaDB memory read and write
│   └── reviewer.py      # Reviewer Agent (Claude)
│
├── memory/
│   └── chroma.py        # ChromaDB -- qa_issues and arch_decisions collections
│
├── tools/
│   ├── search.py        # Tavily search (gracefully disabled if no key)
│   └── file_manager.py  # Saves generated project to output/
│
├── models/
│   └── schemas.py       # Pydantic models (ProductSpec, Architecture, etc.)
│
├── web/
│   ├── app.py           # FastAPI + SSE -- streams LangGraph events to browser
│   └── templates/
│       └── index.html   # Web UI (Tailwind + Alpine.js)
│
├── memory_store/        # ChromaDB data (created on first run)
└── output/              # Generated projects written here
```

## How Memory Works

```
Run 1: "Build a todo API"
  QA finds: "Missing input validation on POST /todos"
  Stored in ChromaDB

Run 2: "Build a notes API"
  QA queries ChromaDB for similar issues
  Retrieves: "Missing input validation on POST /todos"
  Injects into prompt as prior context
  QA catches the pattern earlier and with greater confidence
```

Over multiple runs, the system builds a knowledge base of common mistakes, recurring patterns, and architecture decisions specific to the tech stacks being used.
