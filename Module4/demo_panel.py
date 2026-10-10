"""Module 4 panel for demo.py: turns a Recommendation into what the Gradio widgets show.

Pure Python (no Gradio import), so it is unit-tested with the stub recommender. demo.py only wires
these values into widgets.
"""
from __future__ import annotations

import os

from outfit_board import outfit_board

PER_PIECE_HEADERS = ["piece", "type", "candidates", "selection score", "best ΔH (σ)", "best ΔT (σ)",
                     "clears τ", "notes"]


def _f(x, fmt="{:+.2f}"):
    return "n/a" if x is None else fmt.format(x)


def verdict_markdown(r) -> str:
    b = r.baseline
    if r.verdict == "swap":
        head = f"## Swap your {r.swap_piece['category']}"
    elif r.verdict == "refresh":
        head = "## Already strong: optional trendier options that keep the look"
    else:
        head = "## Already strong, no swap recommended"
    lines = [head, f"Outfit harmony **{b['harmony_pct']:.0f}/100**"
             + (f" · mean piece trend **{b['trend_mean']:.0f}/100**" if b["trend_mean"] is not None else ""),
             f"Trend weight {r.weights['w_t']:.2f} · rule `{r.selection_rule}` · pool `{r.pool_mode}`"]
    if r.flags:
        lines.append("Notes: " + "; ".join(r.flags))
    if r.provisional:
        lines.append("Provisional (not yet calibrated): " + ", ".join(r.provisional))
    return "\n\n".join(lines)


def _change(from_type, to_type) -> str:
    return f"{from_type} → {to_type}" if from_type != to_type else f"{from_type} (same type)"


def _caption(s) -> str:
    trend = ("" if s.trend_before is None or s.trend_after is None
             else f" · trend {s.trend_before:.0f}→{s.trend_after:.0f}")
    return (f"{_change(s.from_type, s.to_type)} · harmony {s.harmony_pct_before:.0f}→{s.harmony_pct_after:.0f}"
            f"{trend} · {s.source}\n{s.what_changed}")


def gallery_items(r, settings) -> list:
    return [(settings.abs(s.crop_path), _caption(s)) for s in r.suggestions]


def also_markdown(r) -> str:
    if not r.also_improve:
        return ""
    parts = [f"{a['piece']} → {a['suggestion'].to_type} (harmony {a['suggestion'].harmony_pct_before:.0f}"
             f"→{a['suggestion'].harmony_pct_after:.0f})" for a in r.also_improve]
    return "**Also worth improving:** " + "; ".join(parts)


def per_piece_rows(r) -> list:
    return [[f, p["type"], p["n_candidates"], _f(p["selection_score"], "{:+.3f}"), _f(p["best_dH"]),
             _f(p["best_dT"]), "✓" if p["clears_tau"] else "", "; ".join(p["flags"])]
            for f, p in r.per_piece.items()]


def choices(r) -> list:
    return list(r.suggestions) + [a["suggestion"] for a in r.also_improve]


def choice_labels(r) -> list:
    top = [f"{i}. {_change(s.from_type, s.to_type)} ({s.source})" for i, s in enumerate(r.suggestions, 1)]
    also = [f"Also: {a['piece']}: {_change(a['suggestion'].from_type, a['suggestion'].to_type)}"
            for a in r.also_improve]
    return top + also


def piece_crop(state, piece, pool, settings):
    if state.outfit_dir:
        return os.path.join(state.outfit_dir, piece.file)
    if piece.pool_row is not None:
        return settings.abs(pool.crop_path(piece.pool_row))
    return None


def outfit_tiles(state, pool, settings) -> list:
    return [(piece_crop(state, p, pool, settings), p.type, False) for p in state.pieces]


def swap_board(r, state, pool, settings, index: int, out_path: str):
    """The outfit with choices(r)[index] swapped in: (board PNG path, one-line summary)."""
    s = choices(r)[index]
    tiles = [(settings.abs(s.crop_path), f"NEW {s.to_type}", True) if p.file == s.piece_file
             else (piece_crop(state, p, pool, settings), p.type, False) for p in state.pieces]
    trend = ("" if s.trend_before is None or s.trend_after is None
             else f" · piece trend {s.trend_before:.0f} → {s.trend_after:.0f}")
    text = (f"{_change(s.from_type, s.to_type)} · Harmony {s.harmony_pct_before:.0f} → "
            f"{s.harmony_pct_after:.0f}/100{trend} · from {s.source}")
    return outfit_board(tiles, out_path), text
