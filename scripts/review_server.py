# ruff: noqa: E501
"""Minimal web UI for human-reviewing synthesis JSONL files.

Two review modes:

1. **Manual** — read every row, press A/R/S. Slow but maximum control.
2. **Verify auto-review** — click "Auto-review" in the sidebar. Heuristics
   mark near-duplicates, refusal artifacts, wrong-length rows, and
   unsafe content as ``rejected`` with a ``rejection_reasons`` list.
   Switch the filter to "rejected" and walk only those rows; press A to
   override any that should have been approved. Pending rows that look
   fine are auto-approved in the same pass.

Keyboard shortcuts
    A — approve (overrides rejection)
    R — reject
    S — skip (leave pending, advance)
    J — previous
    K — next
    G — jump to next pending / rejected (depends on filter)
    F — cycle filter: all → pending → rejected → approved → all

Single-user tool. No auth. Don't expose to the public internet.

Usage::

    python scripts/review_server.py
    python scripts/review_server.py --root data/synthetic/pending --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError as e:
    raise SystemExit("Flask not installed. Run: pip install flask") from e


FILE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Auto-review heuristics
# ---------------------------------------------------------------------------
REFUSAL_PATTERNS = re.compile(
    "|".join([
        r"\bi cannot help (you|with)\b",
        r"\bi can't help (you|with)\b",
        r"\bi cannot assist\b",
        r"\bi can't assist\b",
        r"\bas an ai\b",
        r"\bi'm not able to (help|assist|generate)\b",
        r"\bi am unable to\b",
        r"\bi will not (generate|produce|create)\b",
        r"\bi'm sorry,? but i (cannot|can't|won't|am)\b",
        r"\bi apologize,? but\b",
        r"\bi refuse to\b",
        r"\bagainst my guidelines\b",
        r"\bagainst openai\b",
    ]),
    re.IGNORECASE,
)

SAFETY_GATE_PATTERNS = [
    # Explicit-in-isolation terms that should never appear in our synthesis.
    # The grooming prompt explicitly says "subtle in context", never explicit.
    "penis", "vagina", "blowjob", "masturbat",
    "undress", "naked pic", "nude pic", "sexting", "send nudes",
]


def _normalized_opening(text: str, length: int = 50) -> str:
    t = re.sub(r"\s+", " ", text.lower().strip())
    return t[:length]


def auto_review(rows: list[dict], max_per_cluster: int = 3) -> dict:
    """Apply heuristic rejection rules to every pending row.

    Already-approved or already-rejected rows are left untouched. Returns a
    summary of actions taken.
    """
    # Identify pending rows we're about to decide on.
    pending_indices = [
        i for i, r in enumerate(rows)
        if (r.get("meta") or {}).get("review_status", "pending") == "pending"
    ]

    # Cluster pending rows by their normalized opening phrase
    clusters: dict[str, list[int]] = defaultdict(list)
    for i in pending_indices:
        text = rows[i]["conversation"]["turns"][0]["text"]
        clusters[_normalized_opening(text)].append(i)

    duplicate_drop: set[int] = set()
    for _, idxs in clusters.items():
        for idx in idxs[max_per_cluster:]:
            duplicate_drop.add(idx)

    stats = {
        "total_pending_before": len(pending_indices),
        "approved": 0,
        "rejected": 0,
        "by_reason": defaultdict(int),
    }

    for i in pending_indices:
        row = rows[i]
        turns = row["conversation"]["turns"]
        reasons: list[str] = []

        # Length / turn-count check
        if not (3 <= len(turns) <= 8):
            reasons.append("length_or_turns")
        elif not (20 <= len(turns[-1]["text"]) <= 500):
            reasons.append("length_or_turns")

        # Refusal detection
        full_text = " ".join(t["text"] for t in turns)
        if REFUSAL_PATTERNS.search(full_text):
            reasons.append("refusal")

        # Safety gate — explicit content never expected
        full_lower = full_text.lower()
        if any(pat in full_lower for pat in SAFETY_GATE_PATTERNS):
            reasons.append("safety_gate")

        # Duplicate cluster
        if i in duplicate_drop:
            reasons.append("duplicate_opening")

        row.setdefault("meta", {})
        if reasons:
            row["meta"]["review_status"] = "rejected"
            row["meta"]["rejection_reasons"] = reasons
            row["meta"]["auto_reviewed"] = True
            stats["rejected"] += 1
            for reason in reasons:
                stats["by_reason"][reason] += 1
        else:
            row["meta"]["review_status"] = "approved"
            row["meta"]["auto_reviewed"] = True
            stats["approved"] += 1

    stats["by_reason"] = dict(stats["by_reason"])
    return stats


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAGE synthesis review</title>
  <style>
    :root {
      --bg: #0f1115; --panel: #151922; --panel-2: #1d2230;
      --muted: #7a8390; --fg: #d4d9e2; --accent: #5db2d3;
      --approve: #5bc675; --approve-dim: #2e5f3a;
      --reject: #e0675b; --reject-dim: #6a2e29;
      --warn: #d9a24b; --pending: #7a8390;
      --current-glow: #d9a24b;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: var(--bg); color: var(--fg); height: 100%; }
    body { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }
    aside { background: var(--panel); padding: 16px; overflow-y: auto; border-right: 1px solid #222a3a; }
    main { padding: 20px 28px; overflow-y: auto; }
    h1 { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin: 0 0 12px; }
    .files { list-style: none; padding: 0; margin: 0 0 20px; }
    .files li { padding: 8px 10px; border-radius: 6px; cursor: pointer; margin-bottom: 4px; font-size: 13px; }
    .files li:hover { background: var(--panel-2); }
    .files li.active { background: var(--panel-2); color: var(--accent); }
    .file-counts { display: flex; gap: 6px; margin-top: 4px; font-size: 11px; }
    .chip { padding: 2px 6px; border-radius: 3px; color: #000; font-weight: 500; }
    .chip.pending { background: var(--pending); }
    .chip.approved { background: var(--approve); }
    .chip.rejected { background: var(--reject); }

    .panel-section { margin: 20px 0; padding-top: 12px; border-top: 1px solid #222a3a; }
    .panel-section h1 { margin-bottom: 8px; }
    .btn-block {
      display: block; width: 100%; padding: 8px 12px; border-radius: 6px;
      border: 0; background: var(--panel-2); color: var(--fg);
      font-size: 12px; cursor: pointer; margin-bottom: 6px; text-align: left;
    }
    .btn-block:hover { filter: brightness(1.2); }
    .btn-block.primary { background: var(--accent); color: #051a22; font-weight: 600; }
    .filter-group { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
    .filter-group button { padding: 4px 9px; font-size: 11px; }
    .filter-group button.active { background: var(--accent); color: #051a22; font-weight: 600; }

    header.toolbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
    .progress { height: 4px; background: var(--panel-2); border-radius: 2px; overflow: hidden; margin-bottom: 8px; }
    .progress .bar { height: 100%; background: var(--accent); transition: width 0.2s; }
    .meta-line { font-size: 12px; color: var(--muted); margin-bottom: 14px; }

    .turn { padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; background: var(--panel); font-size: 14px; line-height: 1.45; white-space: pre-wrap; }
    .turn.role-user { border-left: 3px solid #6a8dc1; }
    .turn.role-char { border-left: 3px solid #8f6fd3; }
    .turn.role-system { border-left: 3px solid #7a8390; font-style: italic; color: var(--muted); }
    .turn.target { background: var(--panel-2); box-shadow: 0 0 0 1px var(--current-glow); }
    .turn .role-tag { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
    .target .role-tag { color: var(--current-glow); }

    .labels-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 14px 0; }
    .label-chip { padding: 3px 8px; border-radius: 4px; background: var(--reject); color: #000; font-size: 12px; font-weight: 500; }
    .label-chip.zero { background: #2a3040; color: var(--muted); }

    .review-notes { padding: 10px 14px; background: var(--panel); border-radius: 8px; font-size: 12px; color: var(--muted); margin: 10px 0 16px; border-left: 3px solid var(--warn); }

    .rejection-banner { padding: 10px 14px; background: var(--reject-dim); color: var(--reject); border-radius: 8px; margin: 10px 0; font-size: 13px; }
    .rejection-banner .label { font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; font-size: 11px; margin-right: 6px; }
    .rejection-banner .reasons { display: inline-flex; gap: 6px; flex-wrap: wrap; }
    .rejection-banner .reason-tag { background: rgba(255,255,255,0.08); padding: 1px 6px; border-radius: 3px; font-family: ui-monospace, monospace; font-size: 11px; }

    .actions { display: flex; gap: 10px; margin-top: 16px; position: sticky; bottom: 0; background: var(--bg); padding: 16px 0; flex-wrap: wrap; }
    button { padding: 10px 16px; border-radius: 6px; border: 0; font-size: 13px; font-weight: 600; cursor: pointer; color: #fff; letter-spacing: 0.02em; }
    button.approve { background: var(--approve); color: #071d10; }
    button.reject { background: var(--reject); color: #1d0b08; }
    button.skip, button.nav { background: var(--panel-2); color: var(--fg); }
    button:hover { filter: brightness(1.1); }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    kbd { font-family: ui-monospace, monospace; font-size: 11px; background: rgba(255,255,255,0.1); padding: 1px 4px; border-radius: 3px; margin-left: 6px; }

    .status-banner { padding: 8px 12px; border-radius: 6px; font-size: 13px; margin-bottom: 12px; display: none; }
    .status-banner.approved { background: var(--approve-dim); color: var(--approve); display: block; }
    .status-banner.rejected { background: var(--reject-dim); color: var(--reject); display: block; }

    .source { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); }
    .empty { padding: 40px; text-align: center; color: var(--muted); }
    .hint { font-size: 11px; color: var(--muted); margin-top: 12px; line-height: 1.6; }
    .toast { position: fixed; bottom: 20px; right: 20px; background: var(--panel-2); color: var(--fg); padding: 12px 18px; border-radius: 6px; font-size: 13px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); opacity: 0; transition: opacity 0.2s; pointer-events: none; max-width: 400px; }
    .toast.show { opacity: 1; }
    .modal-bg { position: fixed; inset: 0; background: rgba(0,0,0,0.6); display: none; align-items: center; justify-content: center; z-index: 10; }
    .modal-bg.show { display: flex; }
    .modal { background: var(--panel); padding: 24px; border-radius: 10px; max-width: 480px; width: 90%; }
    .modal h2 { margin: 0 0 12px; font-size: 15px; }
    .modal p { color: var(--muted); font-size: 13px; margin: 8px 0; }
    .modal .buttons { display: flex; gap: 8px; margin-top: 16px; justify-content: flex-end; }
  </style>
</head>
<body>
<aside>
  <h1>Files</h1>
  <ul class="files" id="files"></ul>

  <div class="panel-section">
    <h1>Auto-review</h1>
    <button class="btn-block primary" onclick="autoReview()">Run auto-review</button>
    <div class="hint">
      Applies heuristics to every pending row: drops near-duplicate openings (keep 3/cluster), refusal artifacts, wrong-length or bad-turn rows, and safety-gate terms. Rows that pass all checks are marked approved. You then use the filter to walk only the rejected ones and override if needed.
    </div>
  </div>

  <div class="panel-section">
    <h1>Filter</h1>
    <div class="filter-group" id="filter-group">
      <button data-filter="all"      class="active">all</button>
      <button data-filter="pending">pending</button>
      <button data-filter="rejected">rejected</button>
      <button data-filter="approved">approved</button>
    </div>
    <div class="hint" style="margin-top: 8px;">
      <kbd>F</kbd>cycle
    </div>
  </div>

  <div class="panel-section">
    <h1>Shortcuts</h1>
    <div class="hint">
      <kbd>A</kbd>approve &nbsp; <kbd>R</kbd>reject &nbsp; <kbd>S</kbd>skip<br>
      <kbd>J</kbd>prev &nbsp; <kbd>K</kbd>next &nbsp; <kbd>G</kbd>next in filter
    </div>
  </div>
</aside>

<main id="main">
  <div class="empty">Select a file on the left to begin.</div>
</main>

<div class="toast" id="toast"></div>

<div class="modal-bg" id="modal">
  <div class="modal">
    <h2 id="modal-title">Run auto-review?</h2>
    <p id="modal-body"></p>
    <div class="buttons">
      <button class="skip" onclick="closeModal()">Cancel</button>
      <button class="approve" id="modal-confirm" onclick="confirmModal()">Run</button>
    </div>
  </div>
</div>

<script>
const state = {
  files: [],
  active: null,
  data: null,
  index: 0,
  filter: 'all',
  modalCallback: null,
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
  state.index = _firstInFilter() ?? 0;
  renderFiles();
  renderExample();
}

function _rowStatus(row) {
  return (row.meta && row.meta.review_status) || 'pending';
}

function _matchesFilter(row) {
  if (state.filter === 'all') return true;
  return _rowStatus(row) === state.filter;
}

function _firstInFilter(startFrom = 0) {
  if (!state.data) return null;
  const n = state.data.rows.length;
  for (let i = startFrom; i < n; i++) {
    if (_matchesFilter(state.data.rows[i])) return i;
  }
  return null;
}

function renderExample() {
  const main = document.getElementById('main');
  if (!state.data) {
    main.innerHTML = '<div class="empty">Select a file.</div>';
    return;
  }
  const n = state.data.rows.length;
  if (!(state.index >= 0 && state.index < n)) {
    main.innerHTML = `<div class="empty">No rows match filter "${state.filter}".</div>`;
    return;
  }
  const row = state.data.rows[state.index];
  const status = _rowStatus(row);
  const bar = 100 * (state.index + 1) / n;

  // Count rows in the current filter for the meta line
  let inFilter = 0, idxInFilter = 0;
  for (let i = 0; i < n; i++) {
    if (_matchesFilter(state.data.rows[i])) {
      inFilter += 1;
      if (i <= state.index) idxInFilter += 1;
    }
  }

  const turns = row.conversation.turns.map((t, i) => {
    const isTarget = i === row.conversation.turns.length - 1;
    const cls = `turn role-${t.role}${isTarget ? ' target' : ''}`;
    const tag = isTarget ? `${t.role.toUpperCase()} — CLASSIFICATION TARGET` : t.role.toUpperCase();
    return `<div class="${cls}"><div class="role-tag">${tag}</div><div>${escapeHtml(t.text)}</div></div>`;
  }).join('');

  const labelsRow = Object.entries(row.labels)
    .map(([k, v]) => `<span class="label-chip ${v > 0 ? '' : 'zero'}">${k}: ${v.toFixed(2)}</span>`)
    .join('');

  const notes = row.meta?.generator_notes;
  const rejReasons = row.meta?.rejection_reasons || [];
  const autoReviewed = row.meta?.auto_reviewed;
  const bannerCls = status === 'approved' || status === 'rejected' ? status : '';

  main.innerHTML = `
    <header class="toolbar">
      <div>
        <strong>${state.active}</strong>
        <span class="meta-line" style="margin-left:12px;">
          Filter: <strong>${state.filter}</strong>
          &nbsp;·&nbsp; ${idxInFilter}/${inFilter} in filter
          &nbsp;·&nbsp; row ${state.index + 1} of ${n}
        </span>
      </div>
      <div style="font-size:12px;">
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
      ${autoReviewed ? ` · <span style="color:var(--accent);">auto-reviewed</span>` : ''}
    </div>
    ${rejReasons.length ? `
      <div class="rejection-banner">
        <span class="label">Rejected:</span>
        <span class="reasons">${rejReasons.map(r => `<span class="reason-tag">${r}</span>`).join('')}</span>
      </div>` : ''}
    <div class="status-banner ${bannerCls}">
      ${status === 'approved' ? '✓ Approved' : status === 'rejected' ? '✗ Rejected — press A to override' : ''}
    </div>
    ${notes ? `<div class="review-notes"><strong>Generator notes:</strong> ${escapeHtml(notes)}</div>` : ''}
    <div class="labels-row">${labelsRow || '<span class="label-chip zero">no labels set</span>'}</div>
    ${turns}
    <div class="actions">
      <button class="nav" onclick="navigate(-1)" ${state.index === 0 ? 'disabled' : ''}>◀ prev <kbd>J</kbd></button>
      <button class="approve" onclick="decide('approved')">
        ${status === 'rejected' ? '↻ override → approve' : '✓ approve'} <kbd>A</kbd>
      </button>
      <button class="reject" onclick="decide('rejected')">✗ reject <kbd>R</kbd></button>
      <button class="skip" onclick="decide('pending')">reset to pending <kbd>S</kbd></button>
      <button class="nav" onclick="navigate(1)" ${state.index === n - 1 ? 'disabled' : ''}>next <kbd>K</kbd> ▶</button>
      <button class="nav" onclick="jumpInFilter()">next in filter <kbd>G</kbd></button>
    </div>
  `;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
}

async function decide(status) {
  if (!state.data) return;
  const row = state.data.rows[state.index];
  const prev = _rowStatus(row);
  row.meta = row.meta || {};
  row.meta.review_status = status;
  renderExample();
  const r = await fetch(`/api/file/${encodeURIComponent(state.active)}/row/${state.index}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({review_status: status}),
  });
  if (!r.ok) { toast('save failed'); row.meta.review_status = prev; renderExample(); return; }
  const f = state.files.find(x => x.name === state.active);
  if (f && prev !== status) {
    if (prev in f.counts) f.counts[prev]--;
    if (status in f.counts) f.counts[status]++;
    renderFiles();
  }
  if (status === 'approved' || status === 'rejected') {
    setTimeout(() => jumpInFilter(), 120);
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

function jumpInFilter() {
  if (!state.data) return;
  const n = state.data.rows.length;
  for (let i = state.index + 1; i < n; i++) {
    if (_matchesFilter(state.data.rows[i])) { state.index = i; renderExample(); return; }
  }
  for (let i = 0; i < state.index; i++) {
    if (_matchesFilter(state.data.rows[i])) { state.index = i; renderExample(); return; }
  }
  toast('No other rows match this filter.');
}

function setFilter(f) {
  state.filter = f;
  for (const btn of document.querySelectorAll('#filter-group button')) {
    btn.classList.toggle('active', btn.dataset.filter === f);
  }
  if (state.data && !_matchesFilter(state.data.rows[state.index])) {
    const first = _firstInFilter();
    if (first !== null) state.index = first;
  }
  renderExample();
}

function cycleFilter() {
  const order = ['all', 'pending', 'rejected', 'approved'];
  const next = order[(order.indexOf(state.filter) + 1) % order.length];
  setFilter(next);
}

for (const btn of document.querySelectorAll('#filter-group button')) {
  btn.onclick = () => setFilter(btn.dataset.filter);
}

async function autoReview() {
  if (!state.active) { toast('select a file first'); return; }
  showModal(
    'Run auto-review?',
    `This applies heuristics to every pending row in <strong>${state.active}</strong>: drops near-duplicate openings (keeps 3 per cluster), refusal artifacts, out-of-range lengths, and safety-gate terms. Rows that pass are marked approved.<br><br>Already-approved or already-rejected rows are not touched. You can override any auto-decision afterward.`,
    async () => {
      const r = await fetch(`/api/file/${encodeURIComponent(state.active)}/auto-review`, {method: 'POST'});
      const result = await r.json();
      toast(`auto-review: ${result.approved} approved, ${result.rejected} rejected`);
      await fetchFiles();
      await openFile(state.active);
      setFilter('rejected');  // auto-switch to reviewing the rejects
    }
  );
}

function showModal(title, bodyHtml, onConfirm) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').innerHTML = bodyHtml;
  document.getElementById('modal').classList.add('show');
  state.modalCallback = onConfirm;
}

function closeModal() {
  document.getElementById('modal').classList.remove('show');
  state.modalCallback = null;
}

async function confirmModal() {
  const cb = state.modalCallback;
  closeModal();
  if (cb) await cb();
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (document.getElementById('modal').classList.contains('show')) {
    if (e.key === 'Escape') closeModal();
    return;
  }
  const k = e.key.toLowerCase();
  if (k === 'a') { e.preventDefault(); decide('approved'); }
  else if (k === 'r') { e.preventDefault(); decide('rejected'); }
  else if (k === 's') { e.preventDefault(); decide('pending'); setTimeout(jumpInFilter, 120); }
  else if (k === 'j') { e.preventDefault(); navigate(-1); }
  else if (k === 'k') { e.preventDefault(); navigate(1); }
  else if (k === 'g') { e.preventDefault(); jumpInFilter(); }
  else if (k === 'f') { e.preventDefault(); cycleFilter(); }
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
            out.append({"name": p.name, "counts": _counts(rows)})
        return jsonify(out)

    @app.get("/api/file/<name>")
    def read_file(name: str):
        path = _resolve(name)
        if not path.exists():
            return ("not found", 404)
        rows = _read_jsonl(path)
        return jsonify({"name": name, "rows": rows})

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
            # When user manually overrides, clear auto-reviewer annotations so the
            # rejection banner stops showing on now-approved rows.
            if status == "approved":
                rows[idx]["meta"].pop("rejection_reasons", None)
            _write_jsonl_atomic(path, rows)
            counts = _counts(rows)

        return jsonify({"ok": True, "counts": counts})

    @app.post("/api/file/<name>/auto-review")
    def auto_review_endpoint(name: str):
        path = _resolve(name)
        if not path.exists():
            return ("not found", 404)
        with FILE_LOCK:
            rows = _read_jsonl(path)
            stats = auto_review(rows)
            _write_jsonl_atomic(path, rows)
            counts = _counts(rows)
        return jsonify({"counts": counts, **stats})

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
