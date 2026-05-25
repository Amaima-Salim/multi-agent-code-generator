from agents.base import BaseAgent
from models.schemas import ProductSpec, Architecture, CodeOutput, QAReport, Issue, CodeFile, IssueSeverity

QA_SYSTEM = """\
You are an expert QA Engineer. Review the code carefully and write comprehensive tests.

CRITICAL: Respond with ONLY a valid JSON object.

{
  "issues": [
    {
      "severity": "critical|major|minor|info",
      "file": "filename",
      "line_range": "10-25 or null",
      "description": "what the issue is",
      "suggestion": "how to fix it"
    }
  ],
  "test_files": [
    {
      "filename": "tests/test_something.py",
      "content": "complete pytest file content",
      "language": "python",
      "description": "what these tests cover"
    }
  ],
  "coverage_assessment": "what is tested vs what is missing",
  "overall_score": 7,
  "overall_assessment": "overall code quality summary",
  "critical_issues": ["truly blocking problems only"],
  "strengths": ["things done well"]
}"""

_SEV_MAP = {
    "critical": IssueSeverity.CRITICAL,
    "major": IssueSeverity.MAJOR,
    "minor": IssueSeverity.MINOR,
    "info": IssueSeverity.INFO,
}


class QAAgent(BaseAgent):
    def __init__(self):
        super().__init__("QA Agent", "🧪")

    def run(
        self,
        spec: ProductSpec,
        architecture: Architecture,
        code: CodeOutput,
        memory_context: list[str] | None = None,
    ) -> QAReport:
        self._log(f"Reviewing code… [dim]({self.llm_label})[/dim]")

        if memory_context:
            self._log(
                f"Injecting {len(memory_context)} past issues from ChromaDB memory",
                "dim",
            )

        memory_section = ""
        if memory_context:
            memory_section = (
                "\n\nPAST ISSUES FROM SIMILAR PROJECTS (ChromaDB memory — watch for these patterns):\n"
                + "\n".join(f"- {h}" for h in memory_context)
            )

        content = (
            f"Review this code and write comprehensive tests.\n\n"
            f"PROJECT: {spec.title}\n"
            f"FEATURES: {', '.join(spec.features[:6])}\n"
            f"CRITERIA: {'; '.join(spec.acceptance_criteria[:5])}"
            + memory_section + "\n\n"
            f"CODE FILES:\n{self._code_context(code)}\n\n"
            "Write pytest tests for: happy paths, error cases, edge cases, "
            "input validation. Respond with ONLY the JSON."
        )

        data = self._extract_json(self._call_llm(QA_SYSTEM, [{"role": "user", "content": content}]))
        data["issues"] = [
            Issue(**{**i, "severity": _SEV_MAP.get(i.get("severity", "info"), IssueSeverity.INFO)})
            for i in data.get("issues", []) if isinstance(i, dict)
        ]
        data["test_files"] = [
            CodeFile(**tf) for tf in data.get("test_files", []) if isinstance(tf, dict)
        ]

        report = QAReport(**data)
        crit = sum(1 for i in report.issues if i.severity == IssueSeverity.CRITICAL)
        self._log(
            f"Done — {report.overall_score}/10 · {len(report.issues)} issues ({crit} critical) · "
            f"{len(report.test_files)} test file(s)",
            "green",
        )
        return report

    def _code_context(self, code: CodeOutput) -> str:
        parts = []
        for f in code.files[:8]:
            snippet = f.content[:2500] + "\n[...truncated]" if len(f.content) > 2500 else f.content
            parts.append(f"=== {f.filename} ===\n{snippet}")
        return "\n\n".join(parts)
