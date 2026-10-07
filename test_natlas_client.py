"""Smoke tests for the mock NCAIR1/N-ATLaS client.

Run:
    python test_natlas_client.py
"""

import os

# Ensure the suite is hermetic: the default client must be mock regardless of
# the developer's local environment.
os.environ.setdefault("NATLAS_MODE", "mock")

from natlas_client import MockNatlasClient, _build_prompt, get_client, resolve_client


def test_mock_returns_expected_shape():
    client = MockNatlasClient()
    result = client.generate("Hello", language="en", task="general")

    expected_keys = {
        "mode",
        "output",
        "latency_ms",
        "model",
        "request_payload",
        "response_raw",
        "error",
    }
    assert set(result.keys()) == expected_keys, sorted(result.keys())
    assert result["mode"] == "mock"
    assert result["error"] is None
    assert result["model"] == "NCAIR1/N-ATLaS"
    assert isinstance(result["latency_ms"], int)
    assert result["latency_ms"] >= 0
    # Mock output is always clearly labelled, never presented as real.
    assert "[MOCK / SAMPLE OUTPUT]" in result["output"]
    # Raw response carries mode="mock" (required by the UI / spec).
    assert result["response_raw"].get("mode") == "mock"
    assert result["response_raw"].get("sample") is True


def test_mock_raw_request_has_mode_mock():
    client = MockNatlasClient()
    result = client.generate("How you dey?", language="pcm", task="general")
    assert result["request_payload"].get("mode") == "mock"


def test_mock_output_is_string():
    result = MockNatlasClient().generate("Test", language="en", task="general")
    assert isinstance(result["output"], str) and result["output"].strip()


def test_factory_defaults_to_mock():
    client, mode, warning = resolve_client("mock")
    assert mode == "mock"
    assert warning is None
    assert isinstance(client, MockNatlasClient)


def test_invalid_mode_falls_back_to_mock():
    _, mode, _ = resolve_client("not-a-real-mode")
    assert mode == "mock"


def test_default_client_is_mock():
    assert get_client().mode == "mock"


def test_prompt_uses_llama3_chat_template():
    prompt = _build_prompt("Hello", "en", "general")
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
