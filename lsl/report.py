"""HTML report generation for LSL benchmark results."""
from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .results import project_root


def _results_root_path(results_root: str) -> Path:
    root = Path(results_root)
    if not root.is_absolute():
        root = project_root() / root
    return root


def load_index_rows(results_root: str = "results") -> List[Dict[str, object]]:
    index_path = _results_root_path(results_root) / "index.jsonl"
    rows: List[Dict[str, object]] = []
    if not index_path.exists():
        return rows
    with open(index_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return rows


def _load_payload(path: str) -> Dict[str, object]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = project_root() / file_path
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    return {"payload": payload}


def collect_latest_results(results_root: str = "results") -> List[Dict[str, object]]:
    latest: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in load_index_rows(results_root):
        path = str(row.get("path", ""))
        if not path:
            continue
        payload = _load_payload(path)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        benchmark = str(metadata.get("benchmark") or row.get("benchmark") or payload.get("benchmark") or "unknown")
        dataset = str(metadata.get("dataset") or row.get("dataset") or payload.get("dataset") or "unknown")
        key = (benchmark, dataset)
        latest[key] = {
            "row": row,
            "payload": payload,
            "path": path,
            "benchmark": benchmark,
            "dataset": dataset,
        }
    items = list(latest.values())
    items.sort(key=lambda item: (str(item["benchmark"]), str(item["dataset"]), str(item["row"].get("timestamp", ""))))
    return items


def _extract_status(payload: Dict[str, object]) -> str:
    success = payload.get("success")
    if isinstance(success, bool):
        return "PASS" if success else "FAIL"
    return str(success if success is not None else "UNKNOWN")


def _format_block(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return html.escape(json.dumps(value, indent=2, ensure_ascii=False))
    return html.escape(str(value))


def render_html_report(results_root: str = "results", title: str = "LSL Benchmark Report") -> str:
    items = collect_latest_results(results_root)
    total = len(items)
    passed = sum(1 for item in items if _extract_status(item["payload"]) == "PASS")
    failed = sum(1 for item in items if _extract_status(item["payload"]) == "FAIL")
    benchmark_counts = Counter(str(item["benchmark"]) for item in items)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    rows_html = []
    for item in items:
        payload = item["payload"]
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        summary_parts = []
        for key in ("metrics", "checks", "comparison", "generation", "native_core"):
            if key in payload:
                summary_parts.append(f"<details><summary>{html.escape(key)}</summary><pre>{_format_block(payload[key])}</pre></details>")
        if "sample" in payload:
            summary_parts.append(f"<details><summary>sample</summary><pre>{html.escape(str(payload['sample'])[:800])}</pre></details>")
        if "sample_prompt" in payload:
            summary_parts.append(f"<details><summary>sample_prompt</summary><pre>{html.escape(str(payload['sample_prompt'])[:400])}</pre></details>")
        rows_html.append(
            "<tr>"
            f"<td><code>{html.escape(str(item['benchmark']))}</code></td>"
            f"<td>{html.escape(str(item['dataset']))}</td>"
            f"<td class='status {html.escape(_extract_status(payload).lower())}'>{html.escape(_extract_status(payload))}</td>"
            f"<td>{html.escape(str(metadata.get('timestamp') or item['row'].get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(metadata.get('git_commit') or item['row'].get('git_commit', '')))}</td>"
            f"<td><code>{html.escape(str(item['path']))}</code></td>"
            f"<td>{''.join(summary_parts) if summary_parts else '<em>no metrics payload</em>'}</td>"
            "</tr>"
        )

    benchmark_rows = "".join(
        f"<li><code>{html.escape(name)}</code> x {count}</li>" for name, count in benchmark_counts.most_common()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f9fc;
      --panel: #ffffff;
      --ink: #18212b;
      --muted: #5d6975;
      --line: #d7dfe8;
      --pass: #0f766e;
      --fail: #b91c1c;
      --shadow: 0 10px 28px rgba(24, 33, 43, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, system-ui, sans-serif; }}
    main {{ width: min(1240px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 56px; }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: clamp(32px, 5vw, 58px); line-height: 0.98; margin-bottom: 14px; }}
    h2 {{ font-size: 20px; margin: 0 0 12px; }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 20px 0 28px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: var(--shadow); }}
    .metric {{ font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 8px; }}
    .muted {{ color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); }}
    th, td {{ text-align: left; padding: 12px 10px; vertical-align: top; border-bottom: 1px solid var(--line); font-size: 14px; }}
    th {{ position: sticky; top: 0; background: #f2f6fb; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .status.pass {{ color: var(--pass); font-weight: 700; }}
    .status.fail {{ color: var(--fail); font-weight: 700; }}
    details {{ margin: 8px 0; }}
    details > summary {{ cursor: pointer; color: var(--ink); }}
    pre {{ white-space: pre-wrap; word-break: break-word; margin: 8px 0 0; padding: 10px; border: 1px solid var(--line); border-radius: 6px; background: #fbfdff; }}
    ul {{ margin: 10px 0 0 18px; color: var(--muted); }}
    .section {{ margin-top: 28px; }}
    .note {{ margin-top: 12px; font-size: 13px; color: var(--muted); }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      table, thead, tbody, tr, th, td {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid var(--line); padding: 8px 0; }}
      td {{ border: 0; padding: 8px 10px; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p>Generated at {html.escape(now)} from <code>{html.escape(str(results_root))}</code>.</p>
  <div class="grid">
    <div class="card"><div class="metric">{total}</div><div class="muted">latest benchmark groups</div></div>
    <div class="card"><div class="metric">{passed}</div><div class="muted">groups with PASS status</div></div>
    <div class="card"><div class="metric">{failed}</div><div class="muted">groups with FAIL status</div></div>
    <div class="card"><div class="metric">{len(benchmark_counts)}</div><div class="muted">benchmark names</div></div>
  </div>
  <div class="section card">
    <h2>Benchmark Mix</h2>
    <ul>{benchmark_rows or "<li>No benchmark results found yet.</li>"}</ul>
  </div>
  <div class="section">
    <h2>Latest Results</h2>
    <table>
      <thead>
        <tr>
          <th>Benchmark</th>
          <th>Dataset</th>
          <th>Status</th>
          <th>Timestamp</th>
          <th>Git</th>
          <th>Path</th>
          <th>Details</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html) if rows_html else '<tr><td colspan="7">No results found.</td></tr>'}
      </tbody>
    </table>
  </div>
  <p class="note">This report intentionally preserves the raw metrics payloads so you can inspect proxy/real measurements without losing context.</p>
</main>
</body>
</html>
"""


def write_html_report(output_path: str, results_root: str = "results", title: str = "LSL Benchmark Report") -> str:
    html_text = render_html_report(results_root=results_root, title=title)
    output = Path(output_path)
    if not output.is_absolute():
        output = project_root() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return str(output)

