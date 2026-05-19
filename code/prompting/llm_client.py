import os
import re
from typing import Dict, Tuple

import requests


STRICT_EXECUTE_INSTRUCTION = """

[Strict response rules]
Return the final answer in English.
Do not use markdown code fences.
Do not output Chinese.
The executable final plan must contain exactly one EXECUTE block.
After EXECUTE, each robot must have exactly one line.
Each line must follow: NAME <robot_name> ACTION <valid_action>
If a robot should do nothing, use WAIT.
Use only robot names, object names, and actions provided in the prompt.
"""


DEFAULT_USER_PROMPT = (
    "Based on the task context, action options, and scene description above, "
    "output the next valid robot plan now. Keep object goals fixed; do not swap "
    "objects between target panels. If direct placement is impossible, use a "
    "reachable handoff panel and make unnecessary robots WAIT. Return exactly "
    "one EXECUTE block and no explanation."
)


def _ollama_base_url() -> str:
    """Resolve Ollama HTTP base URL.

    Supports either:
    - OLLAMA_BASE_URL=http://127.0.0.1:11434
    - OLLAMA_HOST=127.0.0.1:11434
    """
    base_url = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST")
    if not base_url:
        base_url = "http://127.0.0.1:11434"
    if not base_url.startswith(("http://", "https://")):
        base_url = "http://" + base_url
    return base_url.rstrip("/")


def clean_llm_response(text: str) -> str:
    """Light cleanup for common local-LLM formatting issues."""
    if text is None:
        return ""
    text = text.strip()

    # Remove markdown fences while preserving their content.
    text = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # If the model emitted multiple EXECUTE blocks, keep from the first one.
    if "EXECUTE" in text:
        text = "EXECUTE" + text.split("EXECUTE", 1)[1]
    return text.strip()


def query_ollama_chat(
    model: str,
    system_prompt: str,
    user_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 300,
) -> Tuple[str, Dict]:
    """Query Ollama's native /api/chat endpoint.

    This avoids installing ollama-python, which can conflict with the
    project's pinned pydantic version.
    """
    url = f"{_ollama_base_url()}/api/chat"
    # Some Ollama chat templates, including Llama-family templates, may stop
    # immediately when the request contains only a system message.  Always send
    # a user turn so the model is explicitly asked to produce the plan.
    messages = [
        {"role": "system", "content": system_prompt + STRICT_EXECUTE_INSTRUCTION},
        {"role": "user", "content": user_prompt.strip() if user_prompt else DEFAULT_USER_PROMPT},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}; body={resp.text[:500]}") from exc
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    usage = {
        "model": data.get("model", model),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
    }
    return clean_llm_response(content), usage
