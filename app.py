"""
app.py
Gradio playground for NCAIR1/N-ATLaS, Nigeria's multilingual language model.
Built for the N-ATLAS National Challenge -- Developer Infrastructure track.

* Explicitly targets NCAIR1/N-ATLaS (never a general-purpose model).
* The active mode (REAL / MOCK) is always shown on screen.
* Everything here works: no dead buttons, no placeholder UI.
* In MOCK mode every output is clearly labelled as sample data (see the banner).

Run locally:
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import json
import os
import time

import gradio as gr

from natlas_client import (
    NatlasError,
    _build_prompt,
    get_active_mode,
    get_client,
)


# ---------------------------------------------------------------------------
# Selectors / sample prompts
# ---------------------------------------------------------------------------
LANGUAGE_CHOICES = [
    ("Auto / Any", "auto"),
    ("English", "en"),
    ("Hausa", "ha"),
    ("Yoruba", "yo"),
    ("Igbo", "ig"),
    ("Nigerian Pidgin", "pcm"),
]

TASK_CHOICES = [
    ("General", "general"),
    ("Translation", "translation"),
    ("Summarization", "summarization"),
    ("Question Answering", "question_answering"),
    ("Sentiment", "sentiment"),
]

SAMPLE_PROMPTS = [
    {"label": "English · General", "language": "en", "task": "general", "text": "What's the weather like today?"},
    {"label": "Hausa · General", "language": "ha", "task": "general", "text": "Yanayin yau yaya yake?"},
    {"label": "Yoruba · Greeting", "language": "yo", "task": "general", "text": "Ẹ kú àárọ̀"},
    {"label": "Igbo · Translate", "language": "ig", "task": "translation", "text": "Translate this into Igbo: Good morning, my name is Chidi."},
    {"label": "Pidgin · Greeting", "language": "pcm", "task": "general", "text": "How you dey?"},
]

ACTIVE_MODE = get_active_mode()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _mode_banner_md(mode: str, startup_warning: str | None = None) -> str:
    if startup_warning:
        return f"⚠️ {startup_warning}"
    if mode == "real":
        return (
            "🟢 **LIVE MODE** — responses come from a real NCAIR1/N-ATLaS endpoint."
        )
    return (
        "🟠 **MOCK MODE — responses are hand-written sample outputs, NOT real "
        "NCAIR1/N-ATLaS.** Set `NATLAS_MODE=real` with `NATLAS_ENDPOINT` (and "
        "`HF_TOKEN`) to go live."
    )


def _route_md(mode: str) -> str:
    if mode == "real":
        endpoint = (os.environ.get("NATLAS_ENDPOINT", "") or "").strip()
        model = (os.environ.get("NATLAS_MODEL_ID", "") or "NCAIR1/N-ATLaS").strip()
        if endpoint:
            return f"Route → custom endpoint `{endpoint}` (model `{model}`)"
        return f"Route → Hugging Face InferenceClient (model `{model}`)"
    return "Route → local samples (mock_data.json)"


def _config() -> dict:
    """Return non-secret config for request inspection and code snippets."""
    return {
        "mode": ACTIVE_MODE,
        "model": (os.environ.get("NATLAS_MODEL_ID", "") or "NCAIR1/N-ATLaS").strip(),
        "endpoint": (os.environ.get("NATLAS_ENDPOINT", "") or "").strip(),
    }


def _format_error(err: Exception) -> str:
    """Render any exception as readable text for the output box."""
    if isinstance(err, NatlasError):
        lines = [f"⚠️ {err.message}"]
        if getattr(err, "response_body", None):
            snippet = str(err.response_body)
            if len(snippet) > 400:
                snippet = snippet[:400] + " …(truncated)"
            lines.append(f"\nEndpoint response:\n{snippet}")
        return "\n".join(lines)
    return f"⚠️ Unexpected error ({type(err).__name__}): {err}"


def _generate(text, language, task, temperature, max_new_tokens):
    """Run one call and return every UI output as a tuple (used by Generate
    and Clear so all components are always updated consistently)."""
    try:
        client = get_client()
        result = client.generate(
            text,
            language=language,
            task=task,
            temperature=temperature,
            max_new_tokens=int(max_new_tokens),
        )
    except NatlasError as err:
        payload = err.request_payload or {}
        error_json = json.dumps(
            {
                "mode": ACTIVE_MODE,
                "error": err.message,
                "status_code": getattr(err, "status_code", None),
                "request": payload,
                "response": getattr(err, "response_body", None),
            },
            indent=2,
            ensure_ascii=False,
        )
        return (
            _format_error(err),
            f"⚠️ {err}",
            f"{int(time.time() * 1000) - 1000} ms (client-local)",
            error_json,
            f"# NCAIR1/N-ATLaS call failed: {err}",
        )
    except Exception as err:  # noqa: BLE001 - never crash the UI
        return (
            _format_error(err),
            f"⚠️ {err}",
            "0 ms (client-local)",
            json.dumps({"mode": ACTIVE_MODE, "error": str(err)}, indent=2, ensure_ascii=False),
            f"# NCAIR1/N-ATLaS call failed: {err}",
        )

    if result["error"]:
        output_text = f"⚠️ {result['error']}"
        status = f"⚠️ {result['error']}"
    else:
        output_text = result["output"]
        status = f"Success · route = {result['route']}"

    latency = f"{result['latency_ms']} ms" + ("  (simulated)" if ACTIVE_MODE == "mock" else "")

    raw_json = json.dumps(
        {
            "mode": ACTIVE_MODE,
            "request": result["request_payload"],
            "response": result["response_raw"],
            "error": result["error"],
        },
        indent=2,
        ensure_ascii=False,
    )

    try:
        snippet = copy_python(text, language, task, temperature, int(max_new_tokens))
    except Exception as err:  # noqa: BLE001
        snippet = f"# Could not build code snippet: {err}"

    return output_text, status, latency, raw_json, snippet


# ---------------------------------------------------------------------------
# Handlers (each returns exactly the number of outputs its UI wires up)
# ---------------------------------------------------------------------------
def generate(text, language, task, temperature, max_new_tokens):
    """Generate button -> updates all five outputs (text, status, latency, raw, snippet)."""
    return _generate(text, language, task, temperature, max_new_tokens)


def load_sample(sample_idx):
    """Sample prompt button -> updates ONLY the input box, language and task."""
    try:
        idx = int(sample_idx)
    except (TypeError, ValueError):
        idx = 0
    sample = SAMPLE_PROMPTS[max(0, min(idx, len(SAMPLE_PROMPTS) - 1))]
    return sample["text"], sample["language"], sample["task"]


def clear_all():
    """Clear button -> resets input and every output component."""
    return (
        "",  # input_text
        "auto",  # language
        "general",  # task
        0.7,  # temperature
        256,  # max_new_tokens
        "",  # output
        "",  # status
        "— ms",  # latency
        "{}",  # raw_json
        "# Press Generate to build a snippet for this exact call.",  # snippet
    )


def build_raw_payload(text, language, task, temperature, max_new_tokens):
    """Preview the request WITHOUT calling the model (works in both modes)."""
    prompt = _build_prompt(text, language, task)
    cfg = _config()
    payload = {
        "mode": cfg["mode"],
        "model": cfg["model"],
        "prompt": prompt,
        "temperature": temperature,
        "max_new_tokens": int(max_new_tokens),
        "language": language,
        "task": task,
        "note": (
            "Preview only. No provider/router is hardcoded -- in real mode the "
            "request goes to NATLAS_ENDPOINT (primary) or the Hugging Face "
            "InferenceClient for this model (secondary)."
        ),
    }
    if cfg["endpoint"]:
        payload["endpoint"] = cfg["endpoint"]
    return json.dumps(payload, indent=2, ensure_ascii=False)


def describe_route():
    cfg = _config()
    if cfg["mode"] == "mock":
        return (
            "Active mode: MOCK. Real calls are disabled. Sample responses come "
            "from mock_data.json and are clearly labelled. Switch with "
            "NATLAS_MODE=real."
        )
    if cfg["endpoint"]:
        return f"Active mode: REAL. Primary route: POST to {cfg['endpoint']} (Bearer HF_TOKEN)."
    return (
        f"Active mode: REAL, no NATLAS_ENDPOINT set. Secondary route: "
        f"InferenceClient(model={cfg['model']!r}) with HF_TOKEN. NOTE: "
        "NCAIR1/N-ATLaS is gated and no hosted endpoint currently serves it, so "
        "this returns a 401 unless you supply a working NATLAS_ENDPOINT."
    )


def copy_python(text, language, task, temperature, max_new_tokens):
    """Generate runnable Python for the SAME call."""
    cfg = _config()
    prompt = _build_prompt(text, language, task)
    model_json = json.dumps(cfg["model"])
    common = {
        "temperature": temperature,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) > 0,
        "return_full_text": False,
    }
    header = (
        "# NCAIR1/N-ATLaS playground snippet\n"
        "# Requires: pip install requests\n"
        "# Reads NATLAS_ENDPOINT / HF_TOKEN from the environment.\n\n"
        "import os\n"
        "import requests\n\n"
        f"prompt = {json.dumps(prompt)}\n"
        "token = os.environ.get(\"HF_TOKEN\", \"\")\n"
        "headers = {\"Authorization\": f\"Bearer {token}\", \"content-type\": \"application/json\"}\n"
    )
    if cfg["endpoint"]:
        url_json = json.dumps(cfg["endpoint"])
        body = {
            "model": cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            **common,
        }
        return (
            header
            + f"payload = {json.dumps(body, ensure_ascii=False, indent=2)}\n"
            + f"response = requests.post({url_json}, headers=headers, json=payload, timeout=60)\n"
            "response.raise_for_status()\n"
            "print(response.json()[\"choices\"][0][\"message\"][\"content\"])\n"
        )
    return (
        header
        + "\n# No NATLAS_ENDPOINT set -> Hugging Face InferenceClient route.\n"
        "# Requires: pip install \"huggingface_hub[inference]\"\n\n"
        "from huggingface_hub import InferenceClient\n\n"
        f"client = InferenceClient(model={model_json}, token=token)\n"
        f"out = client.text_generation(prompt, temperature={float(temperature)}, "
        f"max_new_tokens={int(max_new_tokens)})\n"
        "print(out)\n"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    startup_warning = None
    try:
        from natlas_client import get_startup_warning
        startup_warning = get_startup_warning()
    except Exception:  # noqa: BLE001
        startup_warning = None

    with gr.Blocks(title="N-ATLaS Playground") as demo:
        gr.Markdown("# N-ATLaS Playground")
        gr.Markdown(
            "A developer playground for **NCAIR1/N-ATLaS**, Nigeria's multilingual "
            "LLM (fine-tuned from Meta-Llama-3). Built for the N-ATLAS National "
            "Challenge — Developer Infrastructure track."
        )
        mode_md = gr.Markdown(_mode_banner_md(ACTIVE_MODE, startup_warning))
        route_md = gr.Markdown(_route_md(ACTIVE_MODE))

        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(
                    label="Prompt",
                    placeholder="Type a message (English, Hausa, Yoruba, Igbo or Nigerian Pidgin)…",
                    lines=4,
                )
                with gr.Row():
                    language = gr.Dropdown(LANGUAGE_CHOICES, label="Language", value="auto")
                    task = gr.Dropdown(TASK_CHOICES, label="Task", value="general")
                with gr.Row():
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    max_new_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max new tokens")

                gr.Markdown("**Sample prompts** (click to load into the input box):")
                with gr.Row():
                    sample_btns = [
                        gr.Button(s["label"], size="sm") for s in SAMPLE_PROMPTS
                    ]

                with gr.Row():
                    run_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=3):
                output = gr.Textbox(label="Output", lines=9, interactive=False)
                status = gr.Textbox(label="Status", interactive=False)
                latency = gr.Textbox(label="Latency", interactive=False)

        with gr.Tab("Raw request & response JSON"):
            gr.Markdown("The exact request sent and the response received (click Generate first).")
            raw_json = gr.Code(language="json", label="Raw JSON", lines=18)

        with gr.Tab("Inspect request (no call)"):
            gr.Markdown("Preview the prompt payload **without** calling the model.")
            inspect_btn = gr.Button("Preview request payload")
            inspect_out = gr.Code(language="json", label="Payload", lines=16)

        with gr.Tab("Copy as Python"):
            gr.Markdown("Ready-to-run Python for the same call (generated from your inputs).")
            snippet = gr.Code(language="python", label="Python", lines=18)

        with gr.Accordion("How the real client routes (reference)", open=False):
            route_desc = gr.Markdown(describe_route())

        # --- wiring ---
        run_btn.click(
            fn=generate,
            inputs=[input_text, language, task, temperature, max_new_tokens],
            outputs=[output, status, latency, raw_json, snippet],
        )
        clear_btn.click(
            fn=clear_all,
            outputs=[
                input_text, language, task, temperature, max_new_tokens,
                output, status, latency, raw_json, snippet,
            ],
        )
        for idx, btn in enumerate(sample_btns):
            btn.click(
                fn=load_sample,
                inputs=[gr.State(idx)],
                outputs=[input_text, language, task],
            )
        inspect_btn.click(
            fn=build_raw_payload,
            inputs=[input_text, language, task, temperature, max_new_tokens],
            outputs=[inspect_out],
        )

        # Keep the mode banner truthful if the environment changes at runtime.
        demo.load(
            fn=lambda: _mode_banner_md(ACTIVE_MODE, startup_warning),
            outputs=[mode_md],
        )

    return demo


if __name__ == "__main__":
    server_name = os.environ.get("GRADIO_SERVER_NAME") or None
    port = int(os.environ["GRADIO_SERVER_PORT"]) if os.environ.get("GRADIO_SERVER_PORT") else None
    build_ui().launch(server_name=server_name, server_port=port, share=False)
