from agents.base import BaseAgent
from models.schemas import ProductSpec, CodeOutput, QAReport, ReviewReport, ReviewComment

REVIEWER_SYSTEM = """\
You are a Senior Staff Engineer doing a final code review.

Review for: security vulnerabilities, performance, code quality, maintainability.

CRITICAL: Respond with ONLY a valid JSON object.

{
  "summary": "2-3 sentence overall summary",
  "comments": [
    {
      "file": "filename",
      "category": "security|performance|style|logic|best-practice",
      "comment": "specific observation",
      "suggestion": "how to improve or null",
      "priority": "high|medium|low"
    }
  ],
  "security_assessment": "security review summary",
  "performance_notes": "performance considerations",
  "maintainability_score": 8,
  "approved": true,
  "final_notes": "recommendation to the team"
}"""


class ReviewerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Reviewer Agent", "🔍")

    def run(
        self,
        spec: ProductSpec,
        code: CodeOutput,
        qa_report: QAReport,
    ) -> ReviewReport:
        self._log(f"Final code review… [dim]({self.llm_label})[/dim]")

        qa_summary = (
            f"QA Score: {qa_report.overall_score}/10\n"
            f"Critical: {sum(1 for i in qa_report.issues if i.severity.value == 'critical')}\n"
            f"Total Issues: {len(qa_report.issues)}\n"
            f"Assessment: {qa_report.overall_assessment}"
        )
        code_sample = "\n\n".join(
            f"=== {f.filename} ===\n{f.content[:2000]}" for f in code.files[:5]
        )

        content = (
            f"Final code review for: {spec.title}\n"
            f"Description: {spec.description}\n\n"
            f"QA REPORT:\n{qa_summary}\n\n"
            f"CODE SAMPLE:\n{code_sample}\n\n"
            "Respond with ONLY the JSON."
        )

        data = self._extract_json(self._call_llm(REVIEWER_SYSTEM, [{"role": "user", "content": content}]))
        data["comments"] = [
            ReviewComment(**c) for c in data.get("comments", []) if isinstance(c, dict)
        ]

        report = ReviewReport(**data)
        status = "[green]APPROVED ✅[/green]" if report.approved else "[red]CHANGES REQUESTED ❌[/red]"
        self._log(
            f"Done — {status} · Maintainability {report.maintainability_score}/10",
            "green",
        )
        return report
