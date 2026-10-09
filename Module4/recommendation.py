"""The Recommendation returned by Module 4 (D6): a plain, JSON-serialisable dataclass consumed by
the demo, the slide figure and the evaluation. All numbers are finite floats or None."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Suggestion:
    piece_file: str            # the worn piece this replaces
    pool_row: int
    source: str                # fashion144k | pinterest
    outfit_id: str
    file: str
    crop_path: str             # repo-relative
    from_type: str
    to_type: str
    harmony_pct_before: float
    harmony_pct_after: float
    trend_before: float | None
    trend_after: float | None
    trend_known: bool          # False: ΔT was set to 0 (no confident trend match)
    d_harmony_z: float
    d_trend_z: float
    gain: float
    what_changed: str          # descriptive, not causal


@dataclass
class Recommendation:
    verdict: str               # swap | refresh | keep
    selection_rule: str
    pool_mode: str             # demo | eval
    weights: dict              # {w_h, w_t}
    thresholds: dict           # {tau, eps}
    baseline: dict             # {harmony_pct, harmony_logit, trend_mean}
    swap_piece: dict | None    # {file, category, group} for verdict "swap"
    suggestions: list
    also_improve: list         # [{piece, selection_score, suggestion}]
    per_piece: dict            # file -> {type, group, n_candidates, selection_score, best_gain, ...}
    flags: list
    provisional: list          # settings not yet calibrated

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, s: str) -> "Recommendation":
        d = json.loads(s)
        d["suggestions"] = [Suggestion(**x) for x in d["suggestions"]]
        d["also_improve"] = [{**a, "suggestion": Suggestion(**a["suggestion"])} for a in d["also_improve"]]
        return cls(**d)


def format_text(r: Recommendation) -> str:
    b = r.baseline
    lines = [f"Outfit harmony {b['harmony_pct']:.0f}/100"
             + (f", mean piece trend {b['trend_mean']:.0f}/100" if b["trend_mean"] is not None else "")]
    if r.verdict == "swap":
        lines.append(f"SWAP your {r.swap_piece['category']} ({r.swap_piece['file']})")
    elif r.verdict == "refresh":
        lines.append("ALREADY STRONG. Optional: trendier options that keep the look")
    else:
        lines.append("ALREADY STRONG. No swap recommended")
    for i, s in enumerate(r.suggestions, 1):
        trend = ("" if s.trend_before is None or s.trend_after is None
                 else f"   trend {s.trend_before:.0f} → {s.trend_after:.0f}")
        lines.append(f"  {i}. {s.from_type} → {s.to_type} ({s.piece_file})   harmony "
                     f"{s.harmony_pct_before:.0f} → {s.harmony_pct_after:.0f}{trend}   [{s.source} {s.outfit_id}/{s.file}]")
        lines.append(f"     {s.what_changed}")
    if r.also_improve:
        lines.append("Also worth improving: " + ", ".join(a["piece"] for a in r.also_improve))
    lines.append("Per piece:")
    for f, p in r.per_piece.items():
        sc = "n/a" if p["selection_score"] is None else f"{p['selection_score']:+.3f}"
        lines.append(f"  {f:22s} candidates {p['n_candidates']:>6}  score {sc:>7}  "
                     f"{'✓' if p['clears_tau'] else ' '}  {'; '.join(p['flags'])}")
    if r.flags:
        lines.append("Notes: " + "; ".join(r.flags))
    if r.provisional:
        lines.append("Provisional (not yet calibrated): " + ", ".join(r.provisional))
    return "\n".join(lines)
