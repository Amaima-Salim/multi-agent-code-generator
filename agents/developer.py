"""
Developer Agent — uses OpenAI GPT-4o when available (best at code generation),
falls back to Claude. Optionally enriches each file with Tavily web search.
"""
import re
from agents.base import BaseAgent
from models.schemas import ProductSpec, Architecture, CodeFile, CodeOutput, SetupStep
from tools.search import SearchTool

DEVELOPER_SYSTEM = """\
You are an expert Senior Software Developer. Write clean, production-quality code.

Rules:
- Type hints for all functions (Python)
- Proper error handling and input validation
- Security best practices (no hardcoded secrets, parameterised queries)
- Follow PEP 8

CRITICAL: Output ONLY the raw file content. No markdown fences. No explanations."""

FILE_LIST_SYSTEM = """\
You are planning a project's file structure.
CRITICAL: Respond with ONLY a JSON array.

[
  {
    "filename": "relative/path/file.ext",
    "language": "python|markdown|yaml|dockerfile|text",
    "description": "purpose of this file",
    "priority": 1,
    "search_hint": "optional Tavily search query for this file, or null"
  }
]

Priority 1=core logic, 2=models/routes/utils, 3=config/tests/docs/requirements."""

SETUP_SYSTEM = """\
CRITICAL: Respond with ONLY a JSON object.
{
  "setup_steps": [{"step": 1, "command": "...", "description": "..."}],
  "run_command": "...",
  "test_command": "... or null",
  "notes": "... or null"
}"""


class DeveloperAgent(BaseAgent):
    def __init__(self, search_tool: SearchTool | None = None):
        super().__init__("Developer Agent", "💻")
        self.search = search_tool or SearchTool()
        self.searches_made: list[str] = []

    def run(self, spec: ProductSpec, architecture: Architecture) -> CodeOutput:
        self._log(f"Starting code generation… [dim]({self.llm_label})[/dim]")
        if self.search.enabled:
            self._log("Tavily search enabled — will enrich files with live docs", "dim")

        context = self._build_context(spec, architecture)
        file_plan = self._get_file_plan(context)
        self._log(f"Planned {len(file_plan)} files")

        code_files: list[CodeFile] = []
        for i, info in enumerate(file_plan, 1):
            self._log(f"Writing [{i}/{len(file_plan)}]: [cyan]{info['filename']}[/cyan]")

            # Optional: search for relevant docs before writing this file
            search_ctx = ""
            if self.search.enabled and info.get("search_hint"):
                query = info["search_hint"]
                self.searches_made.append(query)
                docs = self.search.search_for_docs(
                    " ".join(architecture.tech_stack[:3]), query
                )
                if docs:
                    search_ctx = f"\n\nRELEVANT DOCUMENTATION (from web search):\n{docs}"

            content = self._generate_file(context, info, code_files, search_ctx)
            code_files.append(CodeFile(
                filename=info["filename"],
                content=content,
                language=info["language"],
                description=info["description"],
            ))

        setup = self._get_setup(context, code_files)
        self._log(
            f"Done — {len(code_files)} files · run: [cyan]{setup['run_command']}[/cyan]",
            "green",
        )
        return CodeOutput(
            files=code_files,
            setup_steps=setup["setup_steps"],
            run_command=setup["run_command"],
            test_command=setup.get("test_command"),
            notes=setup.get("notes"),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_context(self, spec: ProductSpec, arch: Architecture) -> str:
        return (
            f"PROJECT: {spec.title}\n"
            f"DESCRIPTION: {spec.description}\n\n"
            f"FEATURES:\n" + "\n".join(f"- {f}" for f in spec.features) + "\n\n"
            f"TECH STACK: {', '.join(arch.tech_stack)}\n"
            f"DEPENDENCIES: {', '.join(arch.pip_dependencies)}\n"
            f"ENV VARS: {', '.join(arch.environment_variables) or 'None'}\n\n"
            f"FILE STRUCTURE:\n" + "\n".join(f"- {f.path}: {f.description}" for f in arch.file_structure) + "\n\n"
            f"API DESIGN: {arch.api_design or 'N/A'}\n"
            f"DATABASE: {arch.database_schema or 'N/A'}\n\n"
            f"ACCEPTANCE CRITERIA:\n" + "\n".join(f"- {a}" for a in spec.acceptance_criteria)
        )

    def _get_file_plan(self, context: str) -> list[dict]:
        response = self._call_llm(FILE_LIST_SYSTEM, [{
            "role": "user",
            "content": (
                f"Plan ALL files for this project:\n\n{context}\n\n"
                "Include: app, models, routes, db, config, requirements.txt, "
                ".env.example, README.md, tests. Respond ONLY with the JSON array."
            ),
        }])
        raw = self._extract_json(response)
        files = raw if isinstance(raw, list) else []
        return sorted(files, key=lambda x: x.get("priority", 99))

    def _generate_file(
        self,
        context: str,
        info: dict,
        existing: list[CodeFile],
        search_ctx: str = "",
    ) -> str:
        recent = "\n".join(f"- {f.filename}: {f.description}" for f in existing[-6:]) or "None"
        content = self._call_llm(
            DEVELOPER_SYSTEM,
            [{
                "role": "user",
                "content": (
                    f"Project context:\n{context}\n\n"
                    f"Files already written:\n{recent}"
                    + search_ctx + "\n\n"
                    f"Write the complete content of:\n"
                    f"  Filename: {info['filename']}\n"
                    f"  Language: {info['language']}\n"
                    f"  Purpose:  {info['description']}\n\n"
                    "Output ONLY the raw file content. No markdown fences."
                ),
            }],
            max_tokens=4096,
        )
        return self._strip_fences(content)

    def _get_setup(self, context: str, files: list[CodeFile]) -> dict:
        files_list = "\n".join(f"- {f.filename}" for f in files)
        data = self._extract_json(self._call_llm(
            SETUP_SYSTEM,
            [{"role": "user", "content": f"{context[:500]}\n\nFiles:\n{files_list}\n\nRespond ONLY with JSON."}],
        ))
        data["setup_steps"] = [
            SetupStep(step=i, command=s.get("command", ""), description=s.get("description", ""))
            for i, s in enumerate(data.get("setup_steps", []), 1)
            if isinstance(s, dict)
        ]
        return data

    @staticmethod
    def _strip_fences(content: str) -> str:
        m = re.match(r'^```(?:\w+)?\n(.*?)\n?```$', content.strip(), re.DOTALL)
        return m.group(1) if m else content
