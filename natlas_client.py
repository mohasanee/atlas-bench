"""
natlas_client.py
Unified client for calling NCAIR1/N-ATLaS, Nigeria's multilingual language model.

Exactly one module touches the model. It exposes a common interface:

    BaseNatlasClient.generate(text, language, task, temperature, max_new_tokens)

Implementations
---------------
* RealNatlasClient  -- calls a live NCAIR1/N-ATLaS endpoint.
    * primary   : POST to NATLAS_ENDPOINT (an OpenAI-compatible or TGI URL),
                  authenticated with HF_TOKEN sent as a bearer token.
    * secondary : huggingface_hub.InferenceClient(model="NCAIR1/N-ATLaS"), used
                  only when NATLAS_ENDPOINT is not set.
  There is deliberately NO provider/router hardcoding (no router="together"):
  routing is decided by whichever endpoint/token you configure via env vars.
* MockNatlasClient  -- returns clearly-labelled SAMPLE outputs so the playground
  runs and deploys today even though the real model is gated and no hosted
  NCAIR1/N-ATLaS endpoint is publicly served.

Mode selection
--------------
NATLAS_MODE=real | mock   (anything not exactly "real" resolves to mock, so a
misconfiguration on Hugging Face Spaces degrades to mock instead of crashing).

Configuration is read from environment variables only. Secrets are NEVER
hardcoded.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Dependency-free .env loader (local convenience).
# Real environment variables always win, so Hugging Face Spaces "Variables"
# (and shell exports) take precedence over any local .env file. Values are
# never printed. On Spaces there is no .env (it is git-ignored), so this no-ops.
# ---------------------------------------------------------------------------
def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # A malformed/unreadable .env must never crash the app.
        pass


_load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (from environment only -- never hardcode secrets).
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "NCAIR1/N-ATLaS"

ENV_MODE = os.environ.get("NATLAS_MODE", "mock")
ENV_ENDPOINT = os.environ.get("NATLAS_ENDPOINT", "")
ENV_MODEL_ID = os.environ.get("NATLAS_MODEL_ID", "") or DEFAULT_MODEL_ID
ENV_API_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("NATLAS_TOKEN", "")
ENV_TIMEOUT_S = os.environ.get("NATLAS_TIMEOUT_S", "")
ENV_RETRIES = os.environ.get("NATLAS_RETRIES", "")

# Sensible defaults; overridable via env for slow endpoints.
REQUEST_TIMEOUT_S = float(ENV_TIMEOUT_S) if ENV_TIMEOUT_S else 60.0
MAX_RETRIES = int(ENV_RETRIES) if ENV_RETRIES else 2
RETRY_BACKOFF_S = 0.6


class NatlasError(Exception):
    """Raised for any N-ATLaS call failure, with a human-readable message.

    Carries the raw request/response (when available) so the UI's "raw JSON"
    tab can show exactly what happened.
    """

    def __init__(
        self,
        message: str,
        *,
        request_payload: dict | None = None,
        response_body: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.request_payload = request_payload
        self.response_body = response_body
        self.status_code = status_code


def _looks_like_chatml(prompt: str) -> bool:
    """Detect whether a prompt already uses the Llama-3 ChatML template."""
    return "<|start_header_id|>" in prompt and "<|end_header_id|>" in prompt


def _build_prompt(text: str, language: str | None, task: str | None) -> str:
    """Build a prompt using the Llama-3 ChatML chat template.

    NCAIR1/N-ATLaS is fine-tuned from Meta-Llama-3, so we use its native chat
    template with the correct per-role headers.

    If ``text`` already contains the ChatML markers we honour it as-is, so the
    playground can inspect -- and re-send -- the exact same prompt.
    """
    if _looks_like_chatml(text):
        user_content = text
        system_content = (
            "You are N-ATLaS, a helpful, multilingual assistant for Nigerian "
            "languages."
        )
    else:
        user_content = text.strip()
        hints = []
        if language and language != "auto":
            hints.append(f"Respond in {language}.")
        if task and task != "general":
            hints.append(f"Treat this as a {task} task.")
        hint = " ".join(hints).strip()
        system_content = "You are N-ATLaS, a helpful, multilingual assistant."
        if hint:
            system_content = (
                "You are N-ATLaS, a helpful, multilingual assistant for Nigerian "
                f"languages. {hint}"
            )

    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


class BaseNatlasClient:
    """Common interface. Subclasses must implement ``generate``."""

    mode = "base"

    def generate(
        self,
        text: str,
        *,
        language: str = "auto",
        task: str = "general",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> dict:
        raise NotImplementedError

    # -- helpers shared by implementations ---------------------------------
    @staticmethod
    def _base_result() -> dict:
        return {
            "mode": "",
            "output": "",
            "latency_ms": 0,
            "model": ENV_MODEL_ID,
            "request_payload": {},
            "response_raw": {},
            "route": "",
            "error": None,
        }


class MockNatlasClient(BaseNatlasClient):
    """Return clearly-labelled SAMPLE outputs.

    Every response is prefixed "MOCK / SAMPLE OUTPUT" and the raw JSON carries
    "mode": "mock", so mock text can never be mistaken for real model output.
    """

    mode = "mock"

    def __init__(self, data_path: str | None = None) -> None:
        self.data_path = Path(data_path or Path(__file__).with_name("mock_data.json"))
        self._samples: list[dict] = []
        self._fallback = True
        self._load_data()

    def _load_data(self) -> None:
        try:
            with open(self.data_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._samples = payload.get("samples", [])
            self._fallback = bool(payload.get("fallback", True))
        except (OSError, json.JSONDecodeError, AttributeError):
            # Missing/corrupt data file must not crash the app; use built-ins.
            self._samples = []
            self._fallback = True

    def _pick_sample(self, language: str) -> dict | None:
        lang = (language or "auto").strip().lower()
        if lang in ("", "auto"):
            pool = self._samples
        else:
            pool = [s for s in self._samples if str(s.get("language", "")).lower() == lang]
        if not pool:
            pool = self._samples
        if not pool:
            return None
        return random.choice(pool)

    def generate(
        self,
        text: str,
        *,
        language: str = "auto",
        task: str = "general",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> dict:
        started = time.perf_counter()
        result = self._base_result()
        result["mode"] = "mock"
        result["route"] = "mock"

        prompt = _build_prompt(text, language, task)
        result["request_payload"] = {
            "mode": "mock",
            "model": ENV_MODEL_ID,
            "prompt": prompt,
            "language": language,
            "task": task,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

        # Simulate realistic latency (a short random delay).
        time.sleep(random.uniform(0.15, 0.65))

        sample = self._pick_sample(language)
        if sample is not None:
            body = sample.get("response", "").strip()
            result["output"] = f"[MOCK / SAMPLE OUTPUT]  {body}"
            result["response_raw"] = {
                "mode": "mock",
                "text": body,
                "sample": True,
                "language": sample.get("language", language),
                "task": sample.get("task", task),
                "label": sample.get("label", "sample"),
            }
        else:
            # No mock_data.json at all -- return an explicitly synthetic string.
            result["output"] = (
                "[MOCK / SAMPLE OUTPUT]  Could not load sample data; this is "
                "synthetic filler text generated locally by the playground, NOT "
                "a real N-ATLaS response."
            )
            result["response_raw"] = {"mode": "mock", "text": result["output"], "sample": True}

        result["latency_ms"] = int(round((time.perf_counter() - started) * 1000))
        return result


class RealNatlasClient(BaseNatlasClient):
    """Call a live NCAIR1/N-ATLaS deployment.

    Routing is env-driven and NOT provider-specific:
      * primary   : NATLAS_ENDPOINT  (OpenAI-compatible or TGI URL) + HF_TOKEN bearer
      * secondary : InferenceClient(model="NCAIR1/N-ATLaS") + HF_TOKEN
    """

    mode = "real"

    def __init__(self) -> None:
        self.endpoint = ENV_ENDPOINT.strip()
        self.token = ENV_API_TOKEN.strip()
        self.model = ENV_MODEL_ID.strip() or DEFAULT_MODEL_ID
        self.timeout = REQUEST_TIMEOUT_S
        self.retries = MAX_RETRIES

    # -- configuration ------------------------------------------------------
    def is_configured(self) -> bool:
        """Real mode needs an endpoint, or a token for the HF secondary route."""
        return bool(self.endpoint) or bool(self.token)

    # -- entry point --------------------------------------------------------
    def generate(
        self,
        text: str,
        *,
        language: str = "auto",
        task: str = "general",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> dict:
        started = time.perf_counter()
        result = self._base_result()
        result["mode"] = "real"
        result["model"] = self.model

        prompt = _build_prompt(text, language, task)
        result["request_payload"] = {
            "mode": "real",
            "model": self.model,
            "endpoint": self.endpoint or "(huggingface_hub.InferenceClient)",
            "prompt": prompt,
            "language": language,
            "task": task,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }

        if not self.is_configured():
            result["error"] = (
                "NATLAS_MODE=real but no endpoint is configured. Set "
                "NATLAS_ENDPOINT (recommended) or HF_TOKEN (for the Hugging Face "
                f"InferenceClient route, model={self.model})."
            )
            result["latency_ms"] = int(round((time.perf_counter() - started) * 1000))
            return result

        # primary: custom endpoint if provided, else secondary: HF InferenceClient
        if self.endpoint:
            result["route"] = "endpoint"
            return self._finish(result, self._generate_via_endpoint(result, prompt, temperature, max_new_tokens), started)

        result["route"] = "huggingface"
        return self._finish(result, self._generate_via_huggingface(result, prompt, temperature, max_new_tokens), started)

    def _finish(self, result: dict, outcome: tuple[str, dict], started: float) -> dict:
        output, raw = outcome
        elapsed = int(round((time.perf_counter() - started) * 1000))
        if output is None:
            result["error"] = raw.get("error", "The model call failed.")
            result["response_raw"] = raw
        else:
            result["output"] = output
            result["response_raw"] = raw
        result["latency_ms"] = elapsed
        return result

    # -- primary route: OpenAI-compatible / TGI endpoint --------------------
    def _generate_via_endpoint(self, result: dict, prompt: str, temperature: float, max_new_tokens: int):
        """POST to NATLAS_ENDPOINT. Returns (output_text | None, raw_dict)."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error: dict | None = None
        for attempt in range(self.retries + 1):
            payload, parsed = None, None
            try:
                if self._is_openai_endpoint():
                    payload = self._openai_payload(prompt, temperature, max_new_tokens)
                    parsed = self._post_json(self._chat_url(), headers, payload)
                    output = self._extract_openai(parsed)
                else:
                    payload = self._tgi_payload(prompt, temperature, max_new_tokens)
                    parsed = self._post_json(self._tgi_url(), headers, payload)
                    output = self._extract_tgi(parsed)

                if output is None:
                    return None, {"error": "Unexpected response shape from the endpoint.", "response": parsed}
                return output, parsed
            except NatlasError as exc:
                # 401/403 are terminal -- no point retrying auth failures.
                return None, self._error_detail(exc)
            except requests.Timeout:
                last_error = {
                    "error": (
                        f"Request timed out after {self.timeout:.0f}s "
                        f"(attempt {attempt + 1} of {self.retries + 1})."
                    )
                }
            except requests.ConnectionError:
                last_error = {
                    "error": (
                        f"Could not connect to {self._chat_url() if self._is_openai_endpoint() else self._tgi_url()} "
                        f"(attempt {attempt + 1} of {self.retries + 1}). Check NATLAS_ENDPOINT."
                    )
                }
            except requests.RequestException as exc:
                last_error = {"error": f"Network error: {exc}"}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                return None, {"error": f"Could not parse the response: {exc}", "response": parsed}

            if attempt < self.retries:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))

        return None, last_error or {"error": "The endpoint request failed."}

    def _is_openai_endpoint(self) -> bool:
        return self.endpoint.rstrip("/").endswith("/chat/completions")

    def _chat_url(self) -> str:
        ep = self.endpoint.rstrip("/")
        if ep.endswith("/chat/completions"):
            return ep
        return f"{ep}/chat/completions"

    def _tgi_url(self) -> str:
        ep = self.endpoint.rstrip("/")
        if ep.endswith("/generate") or ep.endswith("/v1"):
            return ep if ep.endswith("/generate") else f"{ep}/generate"
        return f"{ep}/generate"

    def _openai_payload(self, prompt: str, temperature: float, max_new_tokens: int) -> dict:
        # Prompt already carries the Llama-3 ChatML markers -> send it whole.
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }

    def _tgi_payload(self, prompt: str, temperature: float, max_new_tokens: int) -> dict:
        return {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "return_full_text": False,
            },
        }

    def _post_json(self, url: str, headers: dict, payload: dict) -> dict:
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        status = resp.status_code
        if status == 401:
            raise NatlasError(
                "Unauthorized (401): the endpoint rejected HF_TOKEN. If the model "
                "is gated, accept its terms on Hugging Face and use a token from an "
                "account with access.",
                request_payload=payload,
                response_body=resp.text,
                status_code=status,
            )
        if status == 403:
            raise NatlasError(
                "Forbidden (403): this token is not allowed to access N-ATLaS "
                "(gated model, or the inference provider is not enabled for the account).",
                request_payload=payload,
                response_body=resp.text,
                status_code=status,
            )
        if status >= 400:
            raise NatlasError(
                f"The endpoint returned HTTP {status}.",
                request_payload=payload,
                response_body=resp.text,
                status_code=status,
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise NatlasError(
                f"The endpoint returned a non-JSON body: {exc}",
                request_payload=payload,
                response_body=resp.text,
                status_code=status,
            ) from exc

    @staticmethod
    def _extract_openai(parsed: dict) -> str | None:
        try:
            content = parsed["choices"][0]["message"]["content"]
            return content.strip() if isinstance(content, str) else None
        except (KeyError, IndexError, TypeError):
            return None

    @staticmethod
    def _extract_tgi(parsed: dict) -> str | None:
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, dict) and isinstance(first.get("generated_text"), str):
                return first["generated_text"].strip()
        if isinstance(parsed, dict) and isinstance(parsed.get("generated_text"), str):
            return parsed["generated_text"].strip()
        return None

    @staticmethod
    def _error_detail(exc: NatlasError) -> dict:
        return {
            "error": exc.message,
            "status_code": exc.status_code,
            "response": exc.response_body,
        }

    # -- secondary route: huggingface_hub InferenceClient -------------------
    def _generate_via_huggingface(self, result: dict, prompt: str, temperature: float, max_new_tokens: int):
        """Use InferenceClient(model="NCAIR1/N-ATLaS"). Import deferred so the
        dependency is optional and mock/endpoint modes stay light. Returns
        (output_text | None, raw_dict)."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            return None, {
                "error": (
                    "The secondary route needs the optional 'huggingface_hub' "
                    "package. Run: pip install huggingface_hub --upgrade, or set "
                    "NATLAS_ENDPOINT to use the primary route."
                )
            }
        if not self.token:
            return None, {
                "error": (
                    "The Hugging Face InferenceClient route needs HF_TOKEN. Set "
                    "HF_TOKEN, or set NATLAS_ENDPOINT to use the primary route."
                )
            }

        last_error: dict | None = None
        for attempt in range(self.retries + 1):
            try:
                kwargs = {"model": self.model}
                if self.endpoint:
                    kwargs["base_url"] = self.endpoint
                client = InferenceClient(**kwargs, token=self.token)
                text = client.text_generation(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    details=False,
                )
                return text.strip(), {"generated_text": text}
            except Exception as exc:  # noqa: BLE001 - normalise any Hub/inference error
                name = type(exc).__name__
                msg = str(exc)
                lowered = f"{name} {msg}".lower()
                if "401" in msg or "403" in msg or "unauthorized" in lowered or "forbidden" in lowered or "gated" in lowered:
                    return None, {
                        "error": (
                            f"Hugging Face rejected the request ({name}): {msg}. "
                            "The model is gated -- accept its terms on Hugging Face "
                            "and use a token from an account with access. Inference "
                            "Providers may also require a paid/pro account."
                        ),
                        "response": msg,
                    }
                last_error = {"error": f"Hugging Face inference error ({name}): {msg}"}
                if attempt < self.retries:
                    time.sleep(RETRY_BACKOFF_S * (attempt + 1))

        return None, last_error or {"error": "The Hugging Face call failed."}


# ---------------------------------------------------------------------------
# Factory + module-level default client.
# ---------------------------------------------------------------------------
def _resolve_mode(raw_mode: str | None) -> str:
    """Anything other than an explicit 'real' resolves to 'mock' (safe on Spaces)."""
    mode = (raw_mode or "mock").strip().lower()
    return "real" if mode == "real" else "mock"


def resolve_client(mode: str | None = None):
    """Build a client instance, returning (client, resolved_mode, warning).

    * At startup (``mode`` is None): an unset/invalid NATLAS_MODE defaults to
      mock; NATLAS_MODE=real with no working config degrades to mock and sets a
      warning instead of crashing, so Hugging Face Spaces always boots.
    * When the UI explicitly requests real (``mode="real"``), a missing
      config raises NatlasError so the user sees a precise message.
    """
    requested = (mode if mode is not None else ENV_MODE) or "mock"
    chosen = _resolve_mode(requested)

    if chosen == "real":
        candidate = RealNatlasClient()
        if not candidate.is_configured():
            if (mode or "").strip().lower() == "real":
                raise NatlasError(
                    "Real mode was requested, but no endpoint is configured. Set "
                    "NATLAS_ENDPOINT (recommended) or HF_TOKEN for the Hugging Face "
                    f"InferenceClient route (model={candidate.model}); the hosted "
                    "endpoint is gated. See .env.example."
                )
            warning = (
                "NATLAS_MODE=real was set at startup, but no working endpoint is "
                "configured (NATLAS_ENDPOINT or HF_TOKEN). Running in MOCK so the "
                "app stays available."
            )
            return MockNatlasClient(), "mock", warning
        return candidate, "real", None

    return MockNatlasClient(), "mock", None


# Instantiated once at import.
_client, _active_mode, _startup_warning = resolve_client()


def get_client() -> BaseNatlasClient:
    return _client


def get_active_mode() -> str:
    return _active_mode


def get_startup_warning() -> str | None:
    return _startup_warning
