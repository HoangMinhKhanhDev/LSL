"""Local web demo for LSLCoreModel."""
from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional

from .report import write_html_report
from .core import RUNTIME_PROFILE_CHOICES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHITECTURE = ROOT / "docs" / "lsl_architecture.html"


def _bootstrap_namespace(args: argparse.Namespace) -> argparse.Namespace:
    from lsl_chat import bootstrap_checkpoint

    return argparse.Namespace(
        checkpoint=args.checkpoint,
        bootstrap_dataset=args.bootstrap_dataset,
        bootstrap_corpus_path=args.bootstrap_corpus_path,
        bootstrap_tokens=args.bootstrap_tokens,
        bootstrap_chars=args.bootstrap_chars,
        vocab_size=args.vocab_size,
        candidate_cap=args.candidate_cap,
        seed=args.seed,
        lsl_profile=args.lsl_profile,
        no_save_native_upgrade=args.no_save_native_upgrade,
    )


def load_model(args: argparse.Namespace):
    from lsl import LSLCoreModel
    from lsl_chat import ensure_native_chat_path

    checkpoint = os.path.abspath(args.checkpoint)
    if os.path.exists(checkpoint):
        model = LSLCoreModel.load(checkpoint)
        model.set_runtime_profile(args.lsl_profile)
        ensure_native_chat_path(model, checkpoint, save_upgrade=not args.no_save_native_upgrade)
        return model, checkpoint
    if args.no_bootstrap:
        raise FileNotFoundError(checkpoint)
    from lsl_chat import bootstrap_checkpoint

    model = bootstrap_checkpoint(_bootstrap_namespace(args))
    ensure_native_chat_path(model, checkpoint, save_upgrade=not args.no_save_native_upgrade)
    return model, checkpoint


def build_index_html(checkpoint: str, report_path: str, architecture_path: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LSL Local Demo</title>
  <style>
    :root {{
      --bg: #f7f9fc;
      --panel: #ffffff;
      --line: #d7dfe8;
      --ink: #16202a;
      --muted: #56606b;
      --accent: #0f766e;
      --shadow: 0 12px 30px rgba(22, 32, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, system-ui, sans-serif; }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 56px; }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: clamp(34px, 5vw, 68px); line-height: 0.95; }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 18px; margin-top: 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; box-shadow: var(--shadow); }}
    textarea, input {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); padding: 12px; font: inherit; }}
    textarea {{ min-height: 160px; resize: vertical; }}
    button {{ border: 0; border-radius: 6px; padding: 10px 14px; font: inherit; background: var(--accent); color: #fff; cursor: pointer; }}
    button.secondary {{ background: #e6eef5; color: var(--ink); }}
    .row {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 0; padding: 12px; background: #fafcff; border: 1px solid var(--line); border-radius: 6px; min-height: 140px; }}
    .meta {{ margin-top: 14px; display: grid; gap: 8px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .muted {{ color: var(--muted); font-size: 14px; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>LSL Local Demo</h1>
  <p>Chat with one checkpoint, inspect diagnostics, and jump to the architecture map or generated report. Checkpoint: <code>{checkpoint}</code></p>
  <div class="grid">
    <section class="card">
      <h2>Chat</h2>
      <p class="muted">Prompt the model, or use the remember endpoint for explicit facts.</p>
      <div style="margin-top: 12px">
        <textarea id="prompt" placeholder="Ask something..."></textarea>
      </div>
      <div class="row" style="margin-top: 12px">
        <input id="tokens" type="number" min="1" max="512" value="64" style="max-width: 120px" aria-label="Max new tokens">
        <button onclick="sendChat()">Send</button>
        <button class="secondary" onclick="loadDiag()">Refresh diag</button>
      </div>
      <div style="margin-top: 12px">
        <pre id="response">Ready.</pre>
      </div>
    </section>
    <aside class="card">
      <h2>Links</h2>
      <div class="meta">
        <a href="/diag" target="_blank">Diagnostics JSON</a>
        <a href="/report" target="_blank">Latest HTML report</a>
        <a href="/architecture" target="_blank">Architecture map</a>
      </div>
      <div style="margin-top: 18px">
        <h2>Diagnostics</h2>
        <pre id="diag">Loading...</pre>
      </div>
      <div style="margin-top: 18px">
        <h2>Remember</h2>
        <input id="subject" placeholder="subject" style="margin-bottom: 8px">
        <input id="relation" placeholder="relation" style="margin-bottom: 8px">
        <input id="object" placeholder="object" style="margin-bottom: 8px">
        <div class="row">
          <button class="secondary" onclick="remember()">Store fact</button>
        </div>
      </div>
    </aside>
  </div>
</main>
<script>
async function loadDiag() {{
  const resp = await fetch('/api/diag');
  document.getElementById('diag').textContent = await resp.text();
}}
async function sendChat() {{
  const prompt = document.getElementById('prompt').value;
  const max_new_tokens = Number(document.getElementById('tokens').value || 64);
  const resp = await fetch('/api/chat', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{prompt, max_new_tokens}})
  }});
  const data = await resp.json();
  document.getElementById('response').textContent = data.response || JSON.stringify(data, null, 2);
  document.getElementById('diag').textContent = JSON.stringify(data.diagnostics || {{}}, null, 2);
}}
async function remember() {{
  const payload = {{
    subject: document.getElementById('subject').value,
    relation: document.getElementById('relation').value,
    object: document.getElementById('object').value,
  }};
  const resp = await fetch('/api/remember', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(payload)
  }});
  const data = await resp.json();
  document.getElementById('response').textContent = JSON.stringify(data, null, 2);
  loadDiag();
}}
loadDiag();
</script>
</body>
</html>
"""


def make_handler(model, checkpoint: str, report_path: str, architecture_path: str) -> type[BaseHTTPRequestHandler]:
    lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "LSLWebDemo/1.0"

        def _send(self, content: str, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
            data = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: Dict[str, object], status: int = 200) -> None:
            self._send(json.dumps(payload, indent=2, ensure_ascii=False), "application/json; charset=utf-8", status=status)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send(build_index_html(checkpoint, report_path, architecture_path))
                return
            if self.path == "/diag":
                with lock:
                    self._send_json(model.diagnostics())
                return
            if self.path == "/report":
                if not os.path.exists(report_path):
                    write_html_report(report_path)
                if os.path.exists(report_path):
                    self._send(Path(report_path).read_text(encoding="utf-8"))
                else:
                    self._send("Report not available", "text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)
                return
            if self.path == "/architecture":
                if os.path.exists(architecture_path):
                    self._send(Path(architecture_path).read_text(encoding="utf-8"))
                else:
                    self._send("Architecture map not available", "text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)
                return
            self._send("Not found", "text/plain; charset=utf-8", status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                payload = json.loads(body or "{}")
            except json.JSONDecodeError:
                payload = {}
            if self.path == "/api/chat":
                prompt = str(payload.get("prompt", ""))
                max_new_tokens = int(payload.get("max_new_tokens", 64))
                with lock:
                    response = model.answer(prompt)
                    if response is None:
                        response = model.generate(prompt, max_new_tokens=max_new_tokens)
                    diagnostics = model.diagnostics()
                self._send_json({"response": response, "diagnostics": diagnostics})
                return
            if self.path == "/api/remember":
                subject = str(payload.get("subject", "")).strip()
                relation = str(payload.get("relation", "")).strip()
                obj = str(payload.get("object", "")).strip()
                if not subject or not relation or not obj:
                    self._send_json({"error": "subject, relation, and object are required"}, status=400)
                    return
                with lock:
                    model.agent.observe_event(subject, relation, obj, episode_id=int(model.seen_tokens), evidence_id=0)
                    diagnostics = model.diagnostics()
                self._send_json({"ok": True, "diagnostics": diagnostics})
                return
            self._send_json({"error": "not found"}, status=404)

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "lsl_tinystories.json"))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lsl-profile", choices=list(RUNTIME_PROFILE_CHOICES), default="bio_native")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--report-path", type=str, default=os.path.join("results", "lsl_report.html"))
    parser.add_argument("--architecture-path", type=str, default=str(DEFAULT_ARCHITECTURE))
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-dataset", choices=["tinystories", "wikitext2"], default="tinystories")
    parser.add_argument("--bootstrap-corpus-path", type=str, default=None)
    parser.add_argument("--bootstrap-tokens", type=int, default=5000)
    parser.add_argument("--bootstrap-chars", type=int, default=250000)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--candidate-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-native-upgrade", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, checkpoint = load_model(args)
    write_html_report(args.report_path, results_root=args.results_root, title="LSL Benchmark Report")
    handler = make_handler(model, checkpoint, os.path.abspath(args.report_path), os.path.abspath(args.architecture_path))
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"LSL local web demo running at {url}")
    print(f"Checkpoint: {checkpoint}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
