"""TUI UI helpers (CSS styles, patch utilities, session utilities).

Matches the cli_ui.py pattern - only styling and utility functions.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


# ============================================================
# CSS Styles for Textual App
# ============================================================

TUI_CSS = """
Screen {
    layout: vertical;
    background: #1a1a2e;
}

Header {
    background: #16213e;
    color: #e94560;
}

/* Footer widget removed; we use a minimal footer_bar */

#output {
    height: 1fr;
    background: #1a1a2e;
    border: none;
    padding: 1 2;
    scrollbar-gutter: stable;
}

#suggestions {
    height: auto;
    max-height: 10;
    background: #16213e;
    border: tall #0f3460;
    margin: 0 2;
    display: none;
}

#suggestions > ListItem {
    padding: 0 2;
    background: #16213e;
}

#suggestions > ListItem:hover {
    background: #0f3460;
}

#suggestions > ListItem.-highlight {
    background: #e94560;
    color: #fff;
}

#input_area {
    dock: bottom;
    height: 3;
    background: #1a1a2e;
    width: 1fr;
}

#input_line_top, #input_line_bottom {
    height: 1;
    /* Textual CSS 不支持 linear-gradient；渐变线由代码用 Rich Text 渲染 */
    background: #1a1a2e;
    width: 1fr;
    content-align: left middle;
}

#input_row {
    height: 1;
    background: #1a1a2e;
    padding: 0 2;
}

#input_prompt {
    width: 2;
    color: #b388ff;
    text-style: bold;
    content-align: left middle;
}

#input_bar {
    background: #1a1a2e;
    border: none;
    padding: 0;
    height: 1;
}

Input {
    background: #1a1a2e;
    border: none;
}

Input:focus {
    border: none;
}

/* Cursor shape is terminal-dependent; we can only style colors here */
Input > .input--cursor {
    background: #00ffff;
    color: #1a1a2e;
}

/* footer_bar removed */
"""

# ============================================================
# Patch utilities
# ============================================================

PATCH_RE = re.compile(r"\s*\*\*\* Begin Patch[\s\S]*?\*\*\* End Patch", re.MULTILINE)
PATCH_FENCE_RE = re.compile(
    r"```(?:patch|diff|text)?\s*(\*\*\* Begin Patch[\s\S]*?\*\*\* End Patch)\s*```",
    re.MULTILINE,
)


def extract_patch(text: str) -> str | None:
    """Extract patch content from text."""
    m = PATCH_FENCE_RE.search(text)
    if m:
        return m.group(1)
    m = PATCH_RE.search(text)
    return m.group(0).strip() if m else None


def normalize_patch(patch_text: str) -> str:
    """Normalize patch format."""
    lines = patch_text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("Add File:", "Update File:", "Delete File:")) and not stripped.startswith("*** "):
            out.append("*** " + stripped)
            continue
        out.append(line)
    return "\n".join(out)


def patch_requires_confirmation(patch_text: str) -> bool:
    """Check if patch is risky and needs confirmation."""
    if "*** Delete File:" in patch_text:
        return True
    file_ops = patch_text.count("*** Add File:") + patch_text.count("*** Update File:") + patch_text.count("*** Delete File:")
    if file_ops >= 6:
        return True
    changed_lines = 0
    for line in patch_text.splitlines():
        if line.startswith("+") or line.startswith("-"):
            changed_lines += 1
    return changed_lines >= 400


# ============================================================
# Session/Event utilities
# ============================================================

def load_events(log_path: Path) -> list[dict]:
    """Load events from JSONL log file."""
    if not log_path.exists():
        return []
    events: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def parse_ts(ts: str) -> datetime | None:
    """Parse ISO timestamp string."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def summarize_session(events: list[dict]) -> dict:
    """Summarize session statistics from events."""
    stats = {
        "turns": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "llm_calls": 0,
        "llm_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens_est": 0,
        "completion_tokens_est": 0,
        "duration_ms": None,
        "start_ts": None,
        "end_ts": None,
    }
    for e in events:
        et = e.get("type")
        if et == "session_start":
            stats["start_ts"] = e.get("ts")
        elif et == "session_end":
            stats["end_ts"] = e.get("ts")
            stats["turns"] = e.get("turns", stats["turns"])
        elif et == "tool":
            stats["tool_calls"] += 1
            if not e.get("ok", True):
                stats["tool_errors"] += 1
        elif et == "llm":
            stats["llm_calls"] += 1
            if not e.get("ok", True):
                stats["llm_errors"] += 1
            stats["prompt_tokens"] += e.get("prompt_tokens") or 0
            stats["completion_tokens"] += e.get("completion_tokens") or 0
            stats["prompt_tokens_est"] += e.get("prompt_tokens_est") or 0
            stats["completion_tokens_est"] += e.get("completion_tokens_est") or 0

    if stats["start_ts"] and stats["end_ts"]:
        start_dt = parse_ts(stats["start_ts"])
        end_dt = parse_ts(stats["end_ts"])
        if start_dt and end_dt:
            stats["duration_ms"] = int((end_dt - start_dt).total_seconds() * 1000)
    return stats


def export_session(session_id: str, events: list[dict], export_dir: Path) -> Path:
    """Export session data to JSON file."""
    export_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_session(events)
    payload = {"session_id": session_id, "summary": summary, "events": events}
    export_path = export_dir / f"session_{session_id}.json"
    with export_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return export_path
