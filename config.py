from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Primary LLM — pick ONE of these ─────────────────────────────────────
    # xAI Grok (free credits — https://console.x.ai)
    xai_api_key: Optional[str] = None

    # Groq (FREE — https://console.groq.com)
    groq_api_key: Optional[str] = None

    # Anthropic Claude (paid)
    anthropic_api_key: Optional[str] = None

    # OpenAI (paid)
    openai_api_key: Optional[str] = None

    # Google Gemini (FREE tier — https://aistudio.google.com)
    gemini_api_key: Optional[str] = None

    # ── Optional tools ────────────────────────────────────────────────────────
    tavily_api_key: Optional[str] = None

    # ── Model names ───────────────────────────────────────────────────────────
    xai_model: str = "grok-3-mini"
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_model: str = "claude-opus-4-7"
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-1.5-flash"

    max_tokens: int = 8192
    output_dir: str = "output"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── Resolved provider (first available wins) ──────────────────────────────
    @property
    def active_provider(self) -> str:
        if self.xai_api_key:        return "xai"
        if self.groq_api_key:       return "groq"
        if self.anthropic_api_key:  return "anthropic"
        if self.openai_api_key:     return "openai"
        if self.gemini_api_key:     return "gemini"
        raise ValueError(
            "No LLM API key found. Add one to .env:\n"
            "  XAI_API_KEY       (free credits — console.x.ai)\n"
            "  GROQ_API_KEY      (free — console.groq.com)\n"
            "  GEMINI_API_KEY    (free — aistudio.google.com)\n"
            "  ANTHROPIC_API_KEY (paid)\n"
            "  OPENAI_API_KEY    (paid)"
        )

    @property
    def active_model(self) -> str:
        return {
            "xai":       self.xai_model,
            "groq":      self.groq_model,
            "anthropic": self.anthropic_model,
            "openai":    self.openai_model,
            "gemini":    self.gemini_model,
        }[self.active_provider]

    @property
    def active_api_key(self) -> str:
        return {
            "xai":       self.xai_api_key,
            "groq":      self.groq_api_key,
            "anthropic": self.anthropic_api_key,
            "openai":    self.openai_api_key,
            "gemini":    self.gemini_api_key,
        }[self.active_provider]

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)


settings = Settings()
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / settings.output_dir
MEMORY_DIR = BASE_DIR / "memory_store"
OUTPUT_DIR.mkdir(exist_ok=True)
MEMORY_DIR.mkdir(exist_ok=True)
