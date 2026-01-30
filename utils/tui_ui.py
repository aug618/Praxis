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
    background: #0f1115;
}

Header {
    background: #141824;
    color: #e8e8e8;
}

/* Footer widget removed; we use a minimal footer_bar */

#logo {
    height: auto;
    max-height: 100;
    background: #0f1115;
    margin: 1 2 0 2;
    content-align: center middle;
}

#trace {
    height: auto;
    max-height: 12;
    background: #0f1115;
    border: tall #202637;
    margin: 0 2;
    padding: 0 1;
    display: none;
}

#output {
    height: 1fr;
    background: #0f1115;
    border: tall #202637;
    padding: 1 2;
    scrollbar-gutter: stable;
    overflow-y: auto;
}

Collapsible {
    border: tall #202637;
    margin: 0 2;
}

#suggestions {
    height: auto;
    max-height: 10;
    background: #141824;
    border: tall #202637;
    margin: 0 2;
    display: none;
}

#suggestions > ListItem {
    padding: 0 2;
    background: #141824;
}

#suggestions > ListItem:hover {
    background: #202637;
}

#suggestions > ListItem.-highlight {
    background: #4c7dff;
    color: #0f1115;
}

#input_area {
    dock: bottom;
    height: 3;
    background: #0f1115;
    width: 1fr;
}

#input_line_top, #input_line_bottom {
    height: 1;
    /* Textual CSS 不支持 linear-gradient；渐变线由代码用 Rich Text 渲染 */
    background: #0f1115;
    width: 1fr;
    content-align: left middle;
}

#input_row {
    height: 1;
    background: #0f1115;
    padding: 0 2;
}

#input_prompt {
    width: 2;
    color: #7aa2f7;
    text-style: bold;
    content-align: left middle;
}

#input_bar {
    background: #0f1115;
    border: none;
    padding: 0;
    height: 1;
}

Input {
    background: #0f1115;
    border: none;
}

Input:focus {
    border: none;
}

/* Cursor shape is terminal-dependent; we can only style colors here */
Input > .input--cursor {
    background: #00ffff;
    color: #0f1115;
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
