"""Thin model adapter. Swapping provider means changing only this file."""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
import config

_last_call = [0.0]


@dataclass
class LLMResult:
    text: str
    elapsed: float                    # seconds for the model call itself
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached: bool = False
    finish_reason: str | None = None
    retries: int = 0
    error: str | None = None


def parse_json(text):
    """Parse model JSON, tolerating code fences and stray text. Returns (obj, error)."""
    if text is None:
        return None, "empty response"
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    def attempt(x):
        try:
            return json.loads(x), None
        except Exception as e:   # noqa: BLE001
            return None, e

    obj, err = attempt(t)
    if obj is not None:
        return obj, None
    a, b = t.find("{"), t.rfind("}")
    core = t[a:b + 1] if a != -1 and b > a else None
    if core is None:
        return None, "no JSON object found"
    obj, err = attempt(core)
    if obj is not None:
        return obj, None
    # common model slip: trailing commas before } or ]
    obj, err2 = attempt(re.sub(r",(\s*[}\]])", r"\1", core))
    if obj is not None:
        return obj, None
    return None, f"invalid JSON: {err}"


class GeminiClient:
    def __init__(self, model=None, use_cache=True, api_key=None):
        from google import genai
        from google.genai import types
        self.types = types
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env next to app.py).")
        self.client = genai.Client(api_key=key)
        self.model = model or config.MODEL_NAME
        self.use_cache = use_cache
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, system, prompt, max_tokens):
        h = hashlib.sha256(json.dumps([self.model, config.GEN_TEMPERATURE, max_tokens, system, prompt]).encode()).hexdigest()[:24]
        return config.CACHE_DIR / f"{h}.json"

    def generate(self, system, prompt, max_tokens=None):
        max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
        path = self._cache_path(system, prompt, max_tokens)
        if self.use_cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return LLMResult(d["text"], d["elapsed"], d.get("in"), d.get("out"), cached=True, finish_reason=d.get("finish"))
        cfg = self.types.GenerateContentConfig(
            system_instruction=system,
            temperature=config.GEN_TEMPERATURE,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        retries, last_err = 0, None
        for attempt in range(config.MAX_RETRIES):
            gap = config.MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call[0])
            if gap > 0:
                time.sleep(gap)
            t0 = time.time()
            try:
                r = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                elapsed = time.time() - t0
                _last_call[0] = time.time()
                u = r.usage_metadata
                fin = str(r.candidates[0].finish_reason) if r.candidates else None
                res = LLMResult(r.text or "", elapsed, getattr(u, "prompt_token_count", None),
                                getattr(u, "candidates_token_count", None), finish_reason=fin, retries=retries)
                if self.use_cache and res.text:
                    path.write_text(json.dumps(dict(text=res.text, elapsed=elapsed, **{"in": res.input_tokens, "out": res.output_tokens}, finish=fin)), encoding="utf-8")
                return res
            except Exception as e:   # noqa: BLE001
                _last_call[0] = time.time()
                last_err = str(e)
                transient = any(s in last_err for s in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "500", "timed out"))
                if not transient or attempt == config.MAX_RETRIES - 1:
                    break
                retries += 1
                time.sleep(10 * (2 ** attempt))
        return LLMResult("", 0.0, error=last_err, retries=retries)


class MockClient:
    """Offline stand-in so the pipeline and UI can be tested without a key."""
    model = "mock"

    def __init__(self, responder=None):
        self.responder = responder
        self.calls = []

    def generate(self, system, prompt, max_tokens=None):
        self.calls.append((system[:40], len(prompt)))
        txt = self.responder(system, prompt) if self.responder else "{}"
        return LLMResult(txt, 0.01, input_tokens=len(prompt) // 4, output_tokens=len(txt) // 4)
