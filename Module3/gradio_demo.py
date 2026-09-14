"""
gradio_demo.py

Minimal frontend for the trendiness model: upload a photo, get back a
Trendiness Score + which Pinterest style cluster it matched.

IMPORTANT LIMITATION (read this before demoing to your guide):
This scores the WHOLE uploaded photo as one thing, not per-garment. It does
NOT use Prateek's garment segmentation (fashion_segmenter.py) — that
integration doesn't exist yet. It DOES now use his embedding code
(clip_extract_embeddings.py) though, so the CLIP embeddings themselves are
identical to what the rest of the project produces. See README.md, section
"Integrating with the real segmentation pipeline."

Requires clip_extract_embeddings.py in the same folder (or on sys.path).

Usage (Colab or local, wherever trend_model.py + a fitted .pkl already are):
    pip install gradio
    python gradio_demo.py
It launches a local URL; set GRADIO_SHARE=1 (e.g. on Colab) to also get a
public link you can open and demo — off by default so the app isn't exposed.
"""

import os
import tempfile

import gradio as gr

from trend_model import TrendModel, ClipEncoder

# Prefer the real fitted model if present, fall back to the synthetic one
# (for demo/testing purposes only — synthetic scores are NOT meaningful
# fashion data, just proof the wiring works).
MODEL_CANDIDATES = [
    "pinterest_trend_model.pkl",
    "/content/drive/MyDrive/Poster/pinterest_trend_model.pkl",
    "synthetic_trend_model.pkl",
]

model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if model_path is None:
    raise FileNotFoundError(
        "No trend model .pkl found. Looked for: " + ", ".join(MODEL_CANDIDATES) +
        "\nRun build_from_directory() first (see README.md), or run "
        "synthetic_demo.py to generate a demo-only model."
    )

print(f"Loading model from: {model_path}")
model = TrendModel.load(model_path)
encoder = ClipEncoder()
using_synthetic = "synthetic" in model_path


SCALE_BANDS = [
    (0, 10, "Not seen in recent trends", "Barely any recent Pinterest activity for this style — essentially absent from the current moment."),
    (10, 30, "Rarely appearing", "Mostly historical/legacy presence — occasionally seen, but not gaining ground."),
    (30, 50, "Occasional presence", "A steady, minor presence — neither rising nor disappearing."),
    (50, 70, "Actively present", "Consistent, moderate presence in recent activity — a stable, everyday style."),
    (70, 90, "Rising", "Gaining visible traction recently — noticeably more common than it used to be."),
    (90, 101, "Frequently seen right now", "Strong recent surge — this style is showing up a lot in the current period."),
]


def band_for_score(score: float):
    for lo, hi, label, desc in SCALE_BANDS:
        if lo <= score < hi:
            return label, desc
    return SCALE_BANDS[-1][2], SCALE_BANDS[-1][3]


def scale_legend_markdown(current_score: float = None) -> str:
    rows = ["| Range | Meaning | What it indicates |", "|---|---|---|"]
    for lo, hi, label, desc in SCALE_BANDS:
        hi_display = 100 if hi == 101 else hi - 1
        range_str = f"{lo}\u2013{hi_display}"
        is_current = current_score is not None and lo <= current_score < hi
        marker = " \u2190 **this photo**" if is_current else ""
        label_str = f"**{label}**" if is_current else label
        rows.append(f"| {range_str} | {label_str}{marker} | {desc} |")
    return "\n".join(rows)


def score_photo(image):
    if image is None:
        return "Upload a photo first.", "", scale_legend_markdown()

    # Unique temp file per request. A fixed filename would let concurrent
    # uploads clobber each other between save and encode — Gradio serves
    # requests in parallel, and share=True makes concurrency likely.
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="gradio_upload_")
    os.close(fd)
    try:
        image.convert("RGB").save(tmp_path)
        result = model.score_embedding(encoder.encode_images([tmp_path])[0])
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if result["trendiness_score"] is None:
        headline = "No confident style match"
        detail = (
            f"This photo didn't closely match any of the {model.n_clusters} "
            f"style clusters the model learned (best similarity: "
            f"{result['best_similarity']:.2f}). Score withheld rather than "
            f"guessed."
        )
        return headline, detail, scale_legend_markdown()

    score = result["trendiness_score"]
    band_label, band_desc = band_for_score(score)
    headline = f"Trendiness Score: {score:.1f} / 100  —  {band_label}"
    detail = (
        f"{band_desc}\n\n"
        f"Matched style cluster #{result['cluster_id']} "
        f"({result['cluster_size']} similar Pinterest images), "
        f"match confidence {result['similarity']:.2f}."
    )
    if result["low_confidence_cluster"]:
        detail += " Note: this cluster is small, so its score is less reliable."
    return headline, detail, scale_legend_markdown(score)


demo = gr.Interface(
    fn=score_photo,
    inputs=gr.Image(type="pil", label="Upload an outfit photo"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Textbox(label="Details"),
        gr.Markdown(label="What the score means"),
    ],
    title="Trendiness Model — Demo",
    description=(
        ("⚠️ DEMO MODE — running against the SYNTHETIC test model, not real "
         "Pinterest data. Scores here are meaningless. Point this at your "
         "real pinterest_trend_model.pkl before demoing to reviewers.\n\n"
         if using_synthetic else "")
        + "Scores the WHOLE photo as one style (not per-garment — see README "
          "for why). Upload any outfit photo to see which Pinterest style "
          "cluster it's closest to and how trending that cluster currently is."
    ),
)

if __name__ == "__main__":
    # A public Gradio link exposes this app (and its host) to anyone with the
    # URL, so it's opt-in. Set GRADIO_SHARE=1 for a public link (e.g. on Colab).
    share = os.environ.get("GRADIO_SHARE", "0").lower() in ("1", "true", "yes")
    demo.launch(share=share)
