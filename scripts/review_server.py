# ruff: noqa: E501
"""Minimal web UI for human-reviewing synthesis JSONL files.

Lists every JSONL under ``data/synthetic/pending/``, shows one example at a
time as a chat-bubble UI, and lets you approve / reject / skip with either
buttons or keyboard shortcuts (A / R / S). Edits to ``meta.review_status`` are
persisted atomically — temp file + rename — so a crash can't corrupt data.

Usage::

    python scripts/review_server.py
    python scripts/review_server.py --root data/synthetic/pending --port 8765

Then visit http://localhost:8765

Keyboard shortcuts
    A — approve
    R — reject
    S — skip (leave pending, go to next)
    J — previous example
    K — next example
    G — jump to next pending

Single-user tool. No auth. Don't expose to the public internet.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError as e:
    raise SystemExit(
        "Flask not installed. Run: pip install flask"
    ) from e


FILE_LOCK = threading.Lock()

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAGE synthesis review</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #151922;
      --panel-2: #1d2230;
      --muted: #7a8390;
      --fg: #d4d9e2;
      --accent: #5db2d3;
      --approve: #5bc675;
      --approve-dim: #2e5f3a;
      --reject: #e0675b;
      --reject-dim: #6a2e29;
      --warn: #d9a24b;
      --pending: #7a8390;
      --current-glow: #d9a24b;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0;
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
      background: var(--bg); color: var(--fg);
      height: 100%;
    }
    body { display: grid; grid-template-columns: 260px 1fr; height: 100vh; }
    aside {
      background: var(--panel); padding: 16px;
      overflow-y: auto; border-right: 1px solid #222a3a;
    }
    main { padding: 20px 28px; overflow-y: auto; }
    h1 {
      font-size: 14px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--muted); margin: 0 0 12px;
    }
    .files { list-style: none; padding: 0; margin: 0 0 20px; }
    .files li {
      padding: 8px 10px; border-radius: 6px; cursor: pointer;
      margin-bottom: 4px; font-size: 13px;
    }
    .files li:hover { background: var(--panel-2); }
    .files li.active { background: var(--panel-2); color: var(--accent); }
    .file-counts {
      display: flex; gap: 6px; margin-top: 4px; font-size: 11px;
    }
    .chip {
      padding: 2px 6px; border-radius: 3px; color: #000; font-weight: 500;
    }
    .chip.pending { background: var(--pending); }
    .chip.approved { background: var(--approve); }
    .chip.rejected { background: var(--reject); }

    header.toolbar {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 16px;
    }
    .progress {
      height: 4px; background: var(--panel-2); border-radius: 2px;
      overflow: hidden; margin-bottom: 8px;
    }
    .progress .bar { height: 100%; background: var(--accent); transition: width 0.2s; }
    .meta-line { font-size: 12px; color: var(--muted); margin-bottom: 14px; }

    .turn {
      padding: 10px 14px; border-radius: 8px; margin-bottom: 10px;
      background: var(--panel); font-size: 14px; line-height: 1.45;
      white-space: pre-wrap;
    }
    .turn.role-user { border-left: 3px solid #6a8dc1; }
    .turn.role-char { border-left: 3px solid #8f6fd3; }
    .turn.role-system { border-left: 3px solid #7a8390; font-style: italic; color: var(--muted); }
    .turn.target {
      background: var(--panel-2);
      box-shadow: 0 0 0 1px var(--current-glow);
    }
    .turn .role-tag {
      font-size: 11px; color: var(--muted); text-transform: uppercase;
      letter-spacing: 0.08em; margin-bottom: 4px;
    }
    .target .role-tag { color: var(--current-glow); }

    .labels-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 14px 0; }
    .label-chip {
      padding: 3px 8px; border-radius: 4px;
      background: var(--reject); color: #000; font-size: 12px; font-weight: 500;
    }
    .label-chip.zero { background: #2a3040; color: var(--muted); }

    .review-notes {
      padding: 10px 14px; background: var(--panel); border-radius: 8px;
      font-size: 12px; color: var(--muted); margin: 10px 0 16px;
      border-left: 3px solid var(--warn);
    }

    .actions {
      display: flex; gap: 10px; margin-top: 16px;
      position: sticky; bottom: 0; background: var(--bg); padding: 16px 0;
    }
    button {
      padding: 10px 16px; border-radius: 6px; border: 0;
      font-size: 13px; font-weight: 600; cursor: pointer;
      color: #fff; letter-spacing: 0.02em;
    }
    button.approve { background: var(--approve); color: #071d10; }
    button.reject { background: var(--reject); color: #1d0b08; }
    button.skip { background: var(--panel-2); color: var(--fg); }
    button.nav { background: var(--panel-2); color: var(--fg); }
    button:hover { filter: brightness(1.1); }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    kbd {
      font-family: ui-monospace, monospace; font-size: 11px;
      background: rgba(255,255,255,0.1); padding: 1px 4px;
      border-radius: 3px; margin-left: 6px;
    }

    .status-banner {
      padding: 8px 12px; border-radius: 6px; font-size: 13px;
      margin-bottom: 12px; display: none;
    }
    .status-banner.approved { background: var(--approve-dim); color: var(--approve); display: block; }
    .status-banner.rejected { background: var(--reject-dim); color: var(--reject); display: block; }

    .source { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); }
    .empty { padding: 40px; text-align: center; color: var(--muted); }
    .hint { font-size: 11px; color: var(--muted); margin-top: 12px; }
  </style>
</head>
<body>
<aside>
  <h1>Files</h1>
  <ul class="files" id="files"></ul>
  <div class="hint">
    <kbd>A</kbd>approve &nbsp;
    <kbd>R</kbd>reject &nbsp;
    <kbd>S</kbd>skip<br>
    <kbd>J</kbd>prev &nbsp;
    <kbd>K</kbd>next &nbsp;
    <kbd>G</kbd>next pending
  </div>
</aside>
<main id="main">
  <div class="empty">Select a file on the left to begin.</div>
</main>

<script>
const state = {
  files: [],
  active: null,
  data: null,
  index: 0,
};

async function fetchFiles() {
  const r = await fetch('/api/files');
  state.files = await r.json();
  renderFiles();
}

function renderFiles() {
  const ul = document.getElementById('files');
  ul.innerHTML = '';
  for (const f of state.files) {
    const li = document.createElement('li');
    li.className = f.name === state.active ? 'active' : '';
    const total = f.counts.pending + f.counts.approved + f.counts.rejected;
    li.innerHTML = `
      <div>${f.name}</div>
      <div class="file-counts">
        <span class="chip pending">${f.counts.pending}</span>
        <span class="chip approved">${f.counts.approved}</span>
        <span class="chip rejected">${f.counts.rejected}</span>
        <span style="color:var(--muted); font-size:11px;">/ ${total}</span>
      </div>
    `;
    li.onclick = () => openFile(f.name);
    ul.appendChild(li);
  }
}

async function openFile(name) {
  state.active = name;
  const r = await fetch(`/api/file/${encodeURIComponent(name)}`);
  state.data = await r.json();
  state.index = state.data.first_pending ?? 0;
  renderFiles();
  renderExample();
}

function renderExample() {
  const main = document.getElementById('main');
  if (!state.data) {
    main.innerHTML = '<div class="empty">Select a file.</div>';
    return;
  }
  const n = state.data.rows.length;
  const row = state.data.rows[state.index];
  const status = row.meta?.review_status || 'pending';
  const bar = 100 * (state.index + 1) / n;

  const turns = row.conversation.turns.map((t, i) => {
    const isTarget = i === row.conversation.turns.length - 1;
    const cls = `turn role-${t.role}${isTarget ? ' target' : ''}`;
    const tag = isTarget ? `${t.role.toUpperCase()} — CLASSIFICATION TARGET` : t.role.toUpperCase();
    return `<div class="${cls}">
      <div class="role-tag">${tag}</div>
      <div>${escapeHtml(t.text)}</div>
    </div>`;
  }).join('');

  const labelsRow = Object.entries(row.labels)
    .map(([k, v]) => `<span class="label-chip ${v > 0 ? '' : 'zero'}">${k}: ${v.toFixed(2)}</span>`)
    .join('');

  const notes = row.meta?.generator_notes;
  const bannerCls = status === 'approved' || status === 'rejected' ? status : '';

  main.innerHTML = `
    <header class="toolbar">
      <div>
        <strong>${state.active}</strong>
        <span class="meta-line">Example ${state.index + 1} of ${n}</span>
      </div>
      <div style="font-size:12px; color:var(--muted);">
        ${status === 'pending' ? '<span class="chip pending">pending</span>' : ''}
        ${status === 'approved' ? '<span class="chip approved">approved</span>' : ''}
        ${status === 'rejected' ? '<span class="chip rejected">rejected</span>' : ''}
      </div>
    </header>
    <div class="progress"><div class="bar" style="width:${bar}%"></div></div>
    <div class="meta-line">
      <span class="source">source=${row.source || 'unknown'}</span>
      ${row.meta?.category ? ` · category=${row.meta.category}` : ''}
      ${row.meta?.polarity ? ` · polarity=${row.meta.polarity}` : ''}
    </div>
    <div class="status-banner ${bannerCls}">
      ${status === 'approved' ? '✓ Approved' : status === 'rejected' ? '✗ Rejected' : ''}
    </div>
    ${notes ? `<div class="review-notes"><strong>Generator notes:</strong> ${escapeHtml(notes)}</div>` : ''}
    <div class="labels-row">${labelsRow || '<span class="label-chip zero">no labels set</span>'}</div>
    ${turns}
    <div class="actions">
      <button class="nav" onclick="navigate(-1)" ${state.index === 0 ? 'disabled' : ''}>◀ prev <kbd>J</kbd></button>
      <button class="approve" onclick="decide('approved')">✓ approve <kbd>A</kbd></button>
      <button class="reject" onclick="decide('rejected')">✗ reject <kbd>R</kbd></button>
      <button class="skip" onclick="decide('pending'); navigate(1)">skip <kbd>S</kbd></button>
      <button class="nav" onclick="navigate(1)" ${state.index === n - 1 ? 'disabled' : ''}>next <kbd>K</kbd> ▶</button>
      <button class="nav" onclick="jumpPending()">next pending <kbd>G</kbd></button>
    </div>
  `;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
  }[c]));
}

async function decide(status) {
  if (!state.data) return;
  const row = state.data.rows[state.index];
  const prev = row.meta?.review_status;
  row.meta = row.meta || {};
  row.meta.review_status = status;
  // Optimistic UI update
  renderExample();
  // Persist
  const r = await fetch(`/api/file/${encodeURIComponent(state.active)}/row/${state.index}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({review_status: status}),
  });
  if (!r.ok) {
    alert('Save failed; reverting');
    row.meta.review_status = prev;
    renderExample();
    return;
  }
  // Update file counts in sidebar without full reload
  const f = state.files.find(x => x.name === state.active);
  if (f && prev !== status) {
    if (prev in f.counts) f.counts[prev]--;
    if (status in f.counts) f.counts[status]++;
    renderFiles();
  }
  // Auto-advance on approve/reject
  if (status === 'approved' || status === 'rejected') {
    setTimeout(() => navigate(1), 120);
  }
}

function navigate(delta) {
  if (!state.data) return;
  const n = state.data.rows.length;
  const next = Math.max(0, Math.min(n - 1, state.index + delta));
  if (next === state.index) return;
  state.index = next;
  renderExample();
}

function jumpPending() {
  if (!state.data) return;
  const n = state.data.rows.length;
  for (let i = state.index + 1; i < n; i++) {
    if ((state.data.rows[i].meta?.review_status || 'pending') === 'pending') {
      state.index = i; renderExample(); return;
    }
  }
  // Wrap around
  for (let i = 0; i < state.index; i++) {
    if ((state.data.rows[i].meta?.review_status || 'pending') === 'pending') {
      state.index = i; renderExample(); return;
    }
  }
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const k = e.key.toLowerCase();
  if (k === 'a') { e.preventDefault(); decide('approved'); }
  else if (k === 'r') { e.preventDefault(); decide('rejected'); }
  else if (k === 's') { e.preventDefault(); decide('pending'); navigate(1); }
  else if (k === 'j') { e.preventDefault(); navigate(-1); }
  else if (k === 'k') { e.preventDefault(); navigate(1); }
  else if (k === 'g') { e.preventDefault(); jumpPending(); }
});

fetchFiles();
</script>
</body>
</html>
"""


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _counts(rows: list[dict]) -> dict[str, int]:
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for r in rows:
        status = (r.get("meta") or {}).get("review_status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _first_pending(rows: list[dict]) -> int | None:
    for i, r in enumerate(rows):
        if (r.get("meta") or {}).get("review_status", "pending") == "pending":
            return i
    return None


def create_app(root: Path) -> Flask:
    app = Flask(__name__)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def _resolve(name: str) -> Path:
        path = (root / name).resolve()
        if not str(path).startswith(str(root)):
            raise ValueError("path outside review root")
        return path

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.get("/api/files")
    def list_files():
        out = []
        for p in sorted(root.glob("*.jsonl")):
            try:
                rows = _read_jsonl(p)
            except Exception:
                continue
            out.append({
                "name": p.name,
                "counts": _counts(rows),
            })
        return jsonify(out)

    @app.get("/api/file/<name>")
    def read_file(name: str):
        path = _resolve(name)
        if not path.exists():
            return ("not found", 404)
        rows = _read_jsonl(path)
        return jsonify({
            "name": name,
            "rows": rows,
            "first_pending": _first_pending(rows),
        })

    @app.post("/api/file/<name>/row/<int:idx>")
    def update_row(name: str, idx: int):
        path = _resolve(name)
        if not path.exists():
            return ("not found", 404)
        payload = request.get_json(silent=True) or {}
        status = payload.get("review_status")
        if status not in ("pending", "approved", "rejected"):
            return ("bad status", 400)

        with FILE_LOCK:
            rows = _read_jsonl(path)
            if not 0 <= idx < len(rows):
                return ("index out of range", 400)
            rows[idx].setdefault("meta", {})
            rows[idx]["meta"]["review_status"] = status
            _write_jsonl_atomic(path, rows)
            counts = _counts(rows)

        return jsonify({"ok": True, "counts": counts})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for reviewing SAGE synthesis JSONL")
    parser.add_argument("--root", type=Path, default=Path("data/synthetic/pending"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.root)
    print(f"Reviewing JSONL files under: {args.root.resolve()}")
    print(f"Open: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
