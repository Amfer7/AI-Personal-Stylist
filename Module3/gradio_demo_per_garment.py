"""
gradio_demo_per_garment.py

PER-GARMENT trendiness demo — the UI the README flagged as "not yet wired".

Unlike gradio_demo.py (which scores the WHOLE photo as one style), this:
  1. Segments the uploaded photo into individual garments with Prateek's
     fashion_segmenter.py (SegFormer),
  2. Embeds each garment crop with the shared clip_extract_embeddings.py,
  3. Scores EACH garment separately against a PerCategoryTrendModel
     (one trend model per broad category), and
  4. Reports every garment's trendiness, the OUTFIT AVERAGE, and the WEAK LINK
     (the least-trendy piece) — the exact signal Module 4 combines with the
     GNN's harmony/weak-link score.

It also shows the segmented photo with labelled garment boxes so you can see
what was detected.

Requires:
  - a fitted PerCategoryTrendModel .pkl (see build_per_category_from_manifest)
  - clip_extract_embeddings.py + fashion_segmenter.py in the same folder
  - pip install gradio

Usage:
    python gradio_demo_per_garment.py
Set GRADIO_SHARE=1 for a public link (off by default — a public link exposes
the app and its host to anyone with the URL).
"""

import os

import gradio as gr

import fashion_segmenter
import clip_extract_embeddings as cee
from trend_model import PerCategoryTrendModel

# Prefer a locally-fitted model; the Drive path is kept for Colab parity.
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(_HERE, "per_category_trend_model.pkl"),
    os.path.join(_HERE, "..", "PinterestData", "per_category_trend_model.pkl"),
    "/content/drive/MyDrive/Poster/per_category_trend_model.pkl",
]

model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if model_path is None:
    raise FileNotFoundError(
        "No PerCategoryTrendModel .pkl found. Looked for: "
        + ", ".join(MODEL_CANDIDATES)
        + "\nFit one first with build_per_category_from_manifest() on the "
        "Pinterest garment manifest + embeddings (see README.md)."
    )

print(f"Loading per-category trend model from: {model_path}")
model = PerCategoryTrendModel.load(model_path)
KNOWN_CATEGORIES = sorted(model.models.keys())
print(f"Categories available: {KNOWN_CATEGORIES}")


# Same 0-100 interpretation bands as the whole-photo demo, kept local so this
# file stays standalone (importing gradio_demo would run its model loader).
SCALE_BANDS = [
    (0, 10, "Not seen in recent trends", "Barely any recent Pinterest activity for this style."),
    (10, 30, "Rarely appearing", "Mostly historical/legacy presence — not gaining ground."),
    (30, 50, "Occasional presence", "A steady, minor presence — neither rising nor disappearing."),
    (50, 70, "Actively present", "Consistent, moderate presence — a stable, everyday style."),
    (70, 90, "Rising", "Gaining visible traction recently — noticeably more common than before."),
    (90, 101, "Frequently seen right now", "Strong recent surge — showing up a lot in the current period."),
]


def band_for_score(score):
    for lo, hi, label, desc in SCALE_BANDS:
        if lo <= score < hi:
            return label, desc
    return SCALE_BANDS[-1][2], SCALE_BANDS[-1][3]


def scale_legend_markdown(current_score=None) -> str:
    rows = ["| Range | Meaning | What it indicates |", "|---|---|---|"]
    for lo, hi, label, desc in SCALE_BANDS:
        hi_display = 100 if hi == 101 else hi - 1
        range_str = f"{lo}–{hi_display}"
        is_current = current_score is not None and lo <= current_score < hi
        label_str = f"**{label}**" if is_current else label
        marker = " ← **outfit avg**" if is_current else ""
        rows.append(f"| {range_str} | {label_str}{marker} | {desc} |")
    return "\n".join(rows)


def score_outfit_photo(image):
    """Segment -> embed -> per-garment trend score. Returns (annotated_image,
    result_markdown, scale_legend_markdown)."""
    if image is None:
        return None, "Upload a photo first.", scale_legend_markdown()

    img = image.convert("RGB")
    garments = fashion_segmenter.extract_garment_segments(img)
    if not garments:
        return img, ("**No garments detected.** Try a clearer full-body outfit "
                     "photo with the clothing visible."), scale_legend_markdown()

    garment_inputs = []
    for g in garments:
        emb = cee.embed_image(g["image_crop"])
        garment_inputs.append({
            "embedding": emb,
            "category": g["broad_category"],
            "specific_category": g["specific_category"],
            "detection_confidence": g["confidence"],
        })

    result = model.score_outfit(garment_inputs)
    annotated = fashion_segmenter.draw_segformer_overlay(img, garments)

    # Per-garment table
    lines = [
        "### Per-garment trendiness",
        "",
        "| Garment | Category | Detection conf. | Trendiness | Status |",
        "|---|---|---|---|---|",
    ]
    for g in result["garments"]:
        score = g["trendiness_score"]
        is_weak = (result["weak_link_category"] is not None
                   and score is not None
                   and g["category"] == result["weak_link_category"]
                   and score == result["weak_link_score"])
        if score is None:
            reason = g.get("reason", "no confident match")
            score_cell, status = "—", reason
        else:
            band_label, _ = band_for_score(score)
            score_cell = f"{score:.1f}/100"
            status = f"{band_label}{'  ⚠️ weak link' if is_weak else ''}"
        lines.append(
            f"| {g['specific_category']} | {g['category']} | "
            f"{g['detection_confidence']:.2f} | {score_cell} | {status} |"
        )

    lines.append("")
    avg = result["outfit_avg_trendiness"]
    if avg is not None:
        avg_band, avg_desc = band_for_score(avg)
        lines.append(f"**Outfit average trendiness: {avg:.1f}/100 — {avg_band}.** {avg_desc}")
    if result["weak_link_category"]:
        lines.append(
            f"\n**Weak link: `{result['weak_link_category']}` "
            f"({result['weak_link_score']:.1f}/100)** — the piece dragging the "
            f"outfit's trend score down. This is the item Module 4 would target "
            f"for a swap recommendation.")
    if avg is None:
        lines.append(
            "\n_No garment matched a known style cluster confidently, so no "
            "outfit score is shown (withheld rather than guessed)._")

    return annotated, "\n".join(lines), scale_legend_markdown(avg)


demo = gr.Interface(
    fn=score_outfit_photo,
    inputs=gr.Image(type="pil", label="Upload an outfit photo"),
    outputs=[
        gr.Image(type="pil", label="Detected garments"),
        gr.Markdown(label="Per-garment trendiness"),
        gr.Markdown(label="What the score means"),
    ],
    title="Per-Garment Trendiness — Demo",
    description=(
        "Segments the photo into individual garments and scores EACH one against "
        "its own Pinterest trend model (top / bottom / dress / accessory), then "
        "flags the **weak link** — the least on-trend piece. "
        f"Categories this model knows: {', '.join(KNOWN_CATEGORIES)}."
    ),
)

if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0").lower() in ("1", "true", "yes")
    demo.launch(share=share)
