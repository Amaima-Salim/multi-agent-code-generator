import re
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()


class FileManager:
    def save_project(self, state: dict) -> Path:
        from config import OUTPUT_DIR

        spec = state.get("spec") or {}
        slug = self._slugify(spec.get("title", "project"))
        project_dir = OUTPUT_DIR / f"{slug}_{state['project_id']}"
        project_dir.mkdir(parents=True, exist_ok=True)

        files_written = 0
        code = state.get("code") or {}

        for f in code.get("files", []):
            dest = project_dir / f["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f["content"], encoding="utf-8")
            files_written += 1

        # Test files from QA (avoid duplicate filenames)
        qa = state.get("qa_report") or {}
        existing = {f["filename"] for f in code.get("files", [])}
        for tf in qa.get("test_files", []):
            if tf["filename"] not in existing:
                dest = project_dir / tf["filename"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(tf["content"], encoding="utf-8")
                files_written += 1

        # JSON reports
        reports_dir = project_dir / "_reports"
        reports_dir.mkdir(exist_ok=True)
        import json
        for name, key in [
            ("product_spec", "spec"),
            ("architecture", "architecture"),
            ("qa_report", "qa_report"),
            ("review_report", "review"),
        ]:
            obj = state.get(key)
            if obj:
                (reports_dir / f"{name}.json").write_text(
                    json.dumps(obj, indent=2, default=str), encoding="utf-8"
                )

        (project_dir / "PIPELINE_REPORT.md").write_text(
            self._build_report(state), encoding="utf-8"
        )

        console.print(
            f"  [green]💾 Saved {files_written} files → {project_dir}[/green]"
        )
        return project_dir

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "_", text)
        return text[:30].strip("_")

    def _build_report(self, state: dict) -> str:
        spec = state.get("spec") or {}
        arch = state.get("architecture") or {}
        code = state.get("code") or {}
        qa = state.get("qa_report") or {}
        review = state.get("review") or {}

        lines = [
            f"# Pipeline Report: {spec.get('title', 'Project')}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Project ID:** {state['project_id']}",
            f"**Status:** {state.get('status', '?')}",
            "",
            "## Requirement",
            "",
            f"> {state['requirement']}",
            "",
        ]

        if spec:
            lines += [
                "## Product Spec", "",
                f"**{spec.get('title')}** — {spec.get('description', '')}",
                "",
                "**Features:**",
                *[f"- {f}" for f in spec.get("features", [])],
                "",
            ]

        if arch:
            lines += [
                "## Architecture", "",
                arch.get("overview", ""),
                "",
                f"**Tech Stack:** {', '.join(arch.get('tech_stack', []))}",
                "",
                "**Key Decisions:**",
                *[f"- **{d['decision']}**: {d['rationale']}" for d in arch.get("key_decisions", [])],
                "",
            ]

        if code:
            lines += [
                "## Generated Files", "",
                *[f"- `{f['filename']}` — {f['description']}" for f in code.get("files", [])],
                "",
                "## Setup", "",
                *[f"{s['step']}. `{s['command']}` — {s['description']}" for s in code.get("setup_steps", [])],
                "",
                f"**Run:** `{code.get('run_command', '')}`",
                "",
            ]

        if qa:
            lines += [
                "## QA Report", "",
                f"**Score:** {qa.get('overall_score', '?')}/10",
                f"**Assessment:** {qa.get('overall_assessment', '')}",
                "",
                "**Issues:**",
                *[f"- `[{i.get('severity','?').upper()}]` **{i.get('file','?')}**: {i.get('description','')}"
                  for i in qa.get("issues", [])],
                "",
            ]

        if review:
            approved = "✅ Approved" if review.get("approved") else "❌ Changes Requested"
            lines += [
                "## Code Review", "",
                f"**Result:** {approved}",
                f"**Maintainability:** {review.get('maintainability_score', '?')}/10",
                f"**Summary:** {review.get('summary', '')}",
                "",
            ]

        if state.get("memory_hits"):
            lines += [
                "## Memory Hits (ChromaDB)", "",
                *[f"- {h}" for h in state["memory_hits"]],
                "",
            ]

        if state.get("search_queries"):
            lines += [
                "## Web Searches (Tavily)", "",
                *[f"- {q}" for q in state["search_queries"]],
                "",
            ]

        return "\n".join(lines)
