"""
BaseAgent — single interface for Groq, Anthropic, OpenAI, and Gemini.

Provider resolution order (first key found in .env wins):
  Groq → Anthropic → OpenAI → Gemini

Groq uses the OpenAI SDK with a custom base_url, so it's near-identical
to the OpenAI path. Gemini uses google-generativeai SDK.
"""
import json
import re
import time
from rich.console import Console

console = Console()


class BaseAgent:
    def __init__(self, name: str, emoji: str):
        self.name = name
        self.emoji = emoji
        from config import settings
        self._s = settings
        self.provider = settings.active_provider
        self.model = settings.active_model

    @property
    def llm_label(self) -> str:
        return f"{self.provider} / {self.model}"

    # ── Main entry ────────────────────────────────────────────────────────────

    def _call_llm(self, system: str, messages: list, max_tokens: int | None = None) -> str:
        tokens = max_tokens or self._s.max_tokens
        for attempt in range(3):
            try:
                if self.provider in ("groq", "openai", "xai"):
                    return self._call_openai_compat(system, messages, tokens)
                elif self.provider == "anthropic":
                    return self._call_anthropic(system, messages, tokens)
                elif self.provider == "gemini":
                    return self._call_gemini(system, messages, tokens)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
            except Exception as exc:
                # Retry on rate limits
                msg = str(exc).lower()
                if "rate" in msg or "429" in msg or "limit" in msg:
                    wait = 30 if attempt == 0 else 60
                    self._log(f"Rate limited — waiting {wait}s… (attempt {attempt+1}/3)", "yellow")
                    time.sleep(wait)
                elif attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise
        return ""

    # ── Groq + OpenAI (same SDK, different base_url) ─────────────────────────

    def _call_openai_compat(self, system: str, messages: list, tokens: int) -> str:
        from openai import OpenAI

        base_urls = {
            "groq": "https://api.groq.com/openai/v1",
            "xai":  "https://api.x.ai/v1",
        }
        kwargs = {"api_key": self._s.active_api_key}
        if self.provider in base_urls:
            kwargs["base_url"] = base_urls[self.provider]

        client = OpenAI(**kwargs)
        oai_msgs = [{"role": "system", "content": system}] + messages

        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=tokens,
            messages=oai_msgs,
        )
        return resp.choices[0].message.content or ""

    # ── Anthropic ────────────────────────────────────────────────────────────

    def _call_anthropic(self, system: str, messages: list, tokens: int) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._s.active_api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    # ── Google Gemini ─────────────────────────────────────────────────────────

    def _call_gemini(self, system: str, messages: list, tokens: int) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self._s.active_api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
        )
        # Convert OpenAI-style messages to Gemini format
        history = []
        last_user = ""
        for m in messages:
            if m["role"] == "user":
                last_user = m["content"]
            elif m["role"] == "assistant":
                history.append({"role": "model", "parts": [m["content"]]})

        chat = model.start_chat(history=history)
        resp = chat.send_message(
            last_user,
            generation_config={"max_output_tokens": tokens},
        )
        return resp.text

    # ── JSON extraction ───────────────────────────────────────────────────────

    def _extract_json(self, text: str) -> dict | list:
        for pattern in [
            r'```json\s*(.*?)\s*```',
            r'```\s*([\[{].*?[\]}])\s*```',
        ]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        for start, end in [('{', '}'), ('[', ']')]:
            s, e = text.find(start), text.rfind(end) + 1
            if s >= 0 and e > s:
                try:
                    return json.loads(text[s:e])
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"No JSON in response:\n{text[:400]}")

    def _log(self, msg: str, style: str = "blue") -> None:
        console.print(f"  [{style}]{self.emoji} {self.name}:[/{style}] {msg}")
