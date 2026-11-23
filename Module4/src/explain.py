"""Descriptive "what changed" line for a suggestion (D6).

Compares the replacement with the piece it replaces, relative to the rest of the outfit, on the
three cues the GNN's edges carry: formality gap, hue relation and volume. It describes the
difference; it does not claim these are the GNN's reasons (the GNN is non-linear in its inputs).
"""
from __future__ import annotations

import math

FORMALITY_STEP = 5.0   # points on M1's 0-100 formality scale
HUE_STEP = 20.0        # degrees of mean hue gap
VOLUME_RATIO = 1.5     # area ratio that counts as a visible size change


def _hue_gap(h: float, others: list):
    gaps = []
    for o in others:
        if not (math.isnan(h) or math.isnan(o)):
            d = abs(h - o) % 360.0
            gaps.append(d if d <= 180.0 else 360.0 - d)
    return sum(gaps) / len(gaps) if gaps else None


def what_changed(current: dict, candidate: dict, others: list) -> str:
    parts = []
    if candidate["type"] != current["type"]:
        parts.append(f"{current['type']} → {candidate['type']}")
    if others:
        f_rest = [o["formality"] for o in others]
        g0 = sum(abs(current["formality"] - f) for f in f_rest) / len(f_rest)
        g1 = sum(abs(candidate["formality"] - f) for f in f_rest) / len(f_rest)
        if abs(g1 - g0) >= FORMALITY_STEP:
            parts.append(f"formality {'closer to' if g1 < g0 else 'further from'} the rest of the outfit "
                         f"(gap {g0:.0f} → {g1:.0f})")
        h_rest = [o["hue"] for o in others]
        h0, h1 = _hue_gap(current["hue"], h_rest), _hue_gap(candidate["hue"], h_rest)
        if h0 is not None and h1 is not None and abs(h1 - h0) >= HUE_STEP:
            parts.append(f"colour {'closer to' if h1 < h0 else 'more contrast with'} the rest "
                         f"(hue gap {h0:.0f}° → {h1:.0f}°)")
        ratio = (candidate["area"] + 1.0) / (current["area"] + 1.0)
        if ratio >= VOLUME_RATIO:
            parts.append("larger piece relative to the rest")
        elif ratio <= 1.0 / VOLUME_RATIO:
            parts.append("smaller piece relative to the rest")
    if not parts:
        return "Similar formality, colour and volume; the difference is in style details"
    s = "; ".join(parts)
    return s[0].upper() + s[1:]
