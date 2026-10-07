---
title: N-ATLaS Playground
emoji: 🇳🇬
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# atlas-bench

A developer playground for testing **NCAIR1/N-ATLaS**, Nigeria's multilingual AI model, built for the NITDA/NCAIR **N-ATLAS Challenge** (Developer Infrastructure track).

It is a small, real Gradio web app you can run locally and deploy to Hugging Face Spaces. It calls **NCAIR1/N-ATLaS** specifically (not a general-purpose model) and always shows which backend is active.

## What it is

- **`app.py`** — the Gradio playground UI.
- **`natlas_client.py`** — the single place that talks to the model: a `BaseNatlasClient` interface, a `RealNatlasClient`, and a `MockNatlasClient`, chosen by `NATLAS_MODE=real|mock`.
- **`mock_data.json`** — labelled sample outputs for mock mode.
- **`test_natlas_client.py`** — smoke tests for the mock client.
- Config is environment-only (`.env`, never committed). No secrets in code.

## Mock vs Live

The app **ships in MOCK mode by default** so it always runs and deploys. This is an honest demo: mock responses are clearly labelled and are **not** real model output.

| Capability | Status |
| --- | --- |
| UI (input, selectors, sample prompts, output, latency, tabs) | **Working** |
| Raw request/response inspector | **Working** |
| "Copy as Python" snippet | **Working** |
| Mode banner (REAL vs MOCK) | **Working** |
| `RealNatlasClient` code (endpoint + HF route, timeouts, retries, errors) | **Written, not run against a live endpoint** |
| Actual N-ATLaS generations | **Mocked** (sample outputs) |

Why mock for the submission: `NCAIR1/N-ATLaS` on Hugging Face is **gated**, and no public Inference Provider currently serves it, so there is no hosted endpoint to hit without approved access. The real client is complete and ready — switching to live needs only env vars (see below).

In mock mode the UI shows a banner: **"MOCK MODE — responses are hand-written sample outputs, NOT real NCAIR1/N-ATLaS"**, and the raw JSON tab includes `"mode": "mock"`.

## How N-ATLaS is called

`natlas_client.py` builds a prompt using the **Llama-3 ChatML template** (N-ATLaS is fine-tuned from Meta-Llama-3), then routes:

1. **Primary** — `POST` to `NATLAS_ENDPOINT`, which may be:
   - a **TGI** URL (`.../generate`), or
   - an **OpenAI-compatible** URL (`.../chat/completions`).
   Authenticated with `HF_TOKEN` as a bearer token.
2. **Secondary** — if no endpoint is set, `huggingface_hub.InferenceClient(model="NCAIR1/N-ATLaS")`.

There is **no provider/router hardcoding** (no `router="together"`). Routing is decided purely by the env vars you set. Timeouts, retries, and readable errors — including **401/403 gated-access** messages — are handled; nothing crashes the UI.

## How to run locally

Requires Python 3.9+.

```powershell
cd path\to\atlas-bench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# optional: configure
Copy-Item .env.example .env   # then edit .env

python app.py                 # open http://127.0.0.1:7860
python test_natlas_client.py  # run the mock smoke tests
```

## How to use the playground

1. Type a prompt (or click a **sample prompt**).
2. Pick **Language** and **Task**, adjust **Temperature** / **Max tokens** if you like.
3. Press **Generate** — the Output, Status, Latency (ms), Raw JSON tab, and Copy-as-Python tab all update.
4. Use **Inspect request (no call)** to preview the exact payload without calling the model.
5. The banner always shows the active mode (MOCK/LIVE).

### Going live

Set these environment variables (locally in `.env`, or in Spaces → Settings → Variables):

```
NATLAS_MODE=real
NATLAS_ENDPOINT=<your TGI or OpenAI-compatible N-ATLaS URL>
HF_TOKEN=<token with access to the gated model>
```

For the secondary (no-endpoint) route, also install: `pip install "huggingface_hub[inference]"`.

## Deploying to Hugging Face Spaces

The YAML header at the top of this file configures the Space (Gradio SDK, `app.py`). Create a **Space** (SDK: Gradio, App file: `app.py`), push this repo to it, and set `NATLAS_MODE` (and, for live mode, `NATLAS_ENDPOINT` / `HF_TOKEN`) under **Settings → Variables**. `.env` is ignored and must not be committed.

## Beta test results

_(empty for now)_

## Team

_(add team members here)_
