from agents.base import BaseAgent
from models.schemas import ProductSpec, UserStory

PM_SYSTEM = """\
You are an expert Product Manager AI agent. Analyze the requirement and produce a detailed spec.

CRITICAL: Respond with ONLY a valid JSON object. No markdown, no explanations.

{
  "title": "concise project title",
  "description": "2-3 sentence description",
  "features": ["5-10 specific implementable features"],
  "user_stories": [
    {"role": "user type", "action": "what they want", "benefit": "why/value"}
  ],
  "acceptance_criteria": ["5-10 testable criteria"],
  "tech_preferences": "mentioned tech preferences or null",
  "constraints": ["constraints/limitations"],
  "out_of_scope": ["explicitly excluded items"]
}"""


class PMAgent(BaseAgent):
    def __init__(self):
        super().__init__("PM Agent", "📋")

    def run(self, requirement: str) -> ProductSpec:
        self._log(f"Analyzing requirement… [dim]({self.llm_label})[/dim]")

        messages = [{
            "role": "user",
            "content": (
                f"Create a comprehensive product specification:\n\n"
                f"REQUIREMENT: {requirement}\n\n"
                "Be specific and actionable. Respond with ONLY the JSON."
            ),
        }]

        data = self._extract_json(self._call_llm(PM_SYSTEM, messages))
        data["user_stories"] = [
            UserStory(**us) if isinstance(us, dict) else us
            for us in data.get("user_stories", [])
        ]

        spec = ProductSpec(**data)
        self._log(
            f"Done — [bold]{spec.title}[/bold] · {len(spec.features)} features",
            "green",
        )
        return spec
