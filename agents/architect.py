from agents.base import BaseAgent
from models.schemas import ProductSpec, Architecture, ArchFile, TechDecision

ARCHITECT_SYSTEM = """\
You are an expert Software Architect. Design a clean, practical system architecture.

CRITICAL: Respond with ONLY a valid JSON object.

{
  "overview": "architecture overview",
  "tech_stack": ["technologies"],
  "pip_dependencies": ["package>=version"],
  "dev_dependencies": ["pytest>=7.0.0"],
  "file_structure": [
    {"path": "relative/path.py", "file_type": "file", "description": "purpose"}
  ],
  "key_decisions": [
    {"decision": "what", "rationale": "why"}
  ],
  "api_design": "REST API notes or null",
  "database_schema": "schema description or null",
  "environment_variables": ["VAR_NAME"]
}"""


class ArchitectAgent(BaseAgent):
    def __init__(self):
        super().__init__("Architect Agent", "🏗️")

    def run(self, spec: ProductSpec, past_decisions: list[str] | None = None) -> Architecture:
        self._log(f"Designing architecture… [dim]({self.llm_label})[/dim]")

        memory_section = ""
        if past_decisions:
            memory_section = (
                "\n\nPAST ARCHITECTURE DECISIONS (from memory — learn from these):\n"
                + "\n".join(f"- {d}" for d in past_decisions)
            )

        content = (
            f"Design the system architecture for:\n\n"
            f"Title: {spec.title}\n"
            f"Description: {spec.description}\n\n"
            f"Features:\n" + "\n".join(f"- {f}" for f in spec.features) + "\n\n"
            f"Tech Preferences: {spec.tech_preferences or 'None'}\n"
            f"Constraints: {'; '.join(spec.constraints) or 'None'}"
            + memory_section
            + "\n\nRespond with ONLY the JSON."
        )

        data = self._extract_json(self._call_llm(ARCHITECT_SYSTEM, [{"role": "user", "content": content}]))
        data["file_structure"] = [
            ArchFile(**f) if isinstance(f, dict) else f for f in data.get("file_structure", [])
        ]
        data["key_decisions"] = [
            TechDecision(**d) if isinstance(d, dict) else d for d in data.get("key_decisions", [])
        ]

        arch = Architecture(**data)
        self._log(
            f"Done — {len(arch.tech_stack)} technologies · {len(arch.file_structure)} files",
            "green",
        )
        return arch
