"""
demo.py  —  Gradio GUI for the three completed modules (in-process, models stay warm)
=====================================================================================
Upload or paste an outfit photo, choose which module to run, and see the result
figure(s) rendered in the browser:

    Module 1  (perception)     -> segmentation + attributes figure
    Module 2  (harmoniousness) -> end-to-end GNN Harmoniousness score figure
    Module 3  (trend)          -> per-garment Pinterest trendiness + weak link figure
    All                        -> all three figures

Unlike the old CLI, this loads the heavy models (SegFormer + CLIP + the GNN) ONCE
and reuses them for every request — so the *first* run is slow (model load) and
every run after that is fast. It imports the pipeline functions directly instead of
shelling out per click.

Run (from the repo root):
    pip install gradio      # one-time
    python demo.py          # opens the GUI in your browser

Notes:
    - Modules 2 and 3 need Module 1's segmentation output, so picking either runs
      Module 1 first, then scores (Module 3 reuses Module 1's garment embeddings).
    - Module 3 needs a fitted Module3/per_category_trend_model.pkl
      (python Module3/fit_per_category.py).
    - First Module-2 run builds a small reference cache (demos/_ref_scores.json);
      later runs load it instantly.
    - Assumes the trained checkpoint + feature store are present under Fashion144k_v1/.
"""

import os
import sys
import time
import importlib.util

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
MOD1 = os.path.join(HERE, "Module1")
MOD2 = os.path.join(HERE, "Module2")
MOD3 = os.path.join(HERE, "Module3")
# Module1/run_pipeline.py imports its siblings by bare name; make sure they resolve.
if MOD1 not in sys.path:
    sys.path.insert(0, MOD1)
# Module3 is appended (not prepended) so Module1's originals win for the vendored
# names they share; the trend pickle needs `trend_model` importable by bare name.
if MOD3 not in sys.path:
    sys.path.append(MOD3)

DATA = os.path.join(HERE, "Fashion144k_v1")
DEMO_OUT = os.path.join(DATA, "demo_out")
DEMOS = os.path.join(HERE, "demos")
CKPT = os.path.join(DATA, "ck_attr_subset_s42", "clip_outfit_gnn_best.pt")
REF_CACHE = os.path.join(DEMOS, "_ref_scores.json")
TREND_MODEL = os.path.join(MOD3, "per_category_trend_model.pkl")

IMAGE_ID = "demo_live"
EXAMPLE_PHOTO = os.path.join(HERE, "samples", "image2.jpg")

MODULE_1 = "Module 1 (perception)"
MODULE_2 = "Module 2 (harmoniousness)"
MODULE_3 = "Module 3 (trend)"
ALL = "All"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- lazy, load-once singletons (heavy models) ----
_pipeline = None   # (run_single_image, render)
_scorer = None     # demo_score.PhotoScorer
_trend = None      # (PerCategoryTrendModel, render_trend_demo.render)


def _get_pipeline():
    """Module 1: run_single_image() + the perception-figure render(). The underlying
    SegFormer/CLIP models cache themselves in their own module globals, so importing
    once means they load once."""
    global _pipeline
    if _pipeline is None:
        rp = _load_module(os.path.join(MOD1, "run_pipeline.py"), "m1_run_pipeline")
        rd = _load_module(os.path.join(MOD1, "render_perception_demo.py"), "m1_render")
        _pipeline = (rp.run_single_image, rd.render)
    return _pipeline


def _get_scorer():
    """Module 2: PhotoScorer loads the GNN + dataset + reference distribution once."""
    global _scorer
    if _scorer is None:
        ds_mod = _load_module(os.path.join(MOD2, "demo_score.py"), "m2_demo_score")
        _scorer = ds_mod.PhotoScorer(checkpoint=CKPT, ref_cache=REF_CACHE)
    return _scorer


def _get_trend():
    """Module 3: the fitted PerCategoryTrendModel + its figure renderer (load once)."""
    global _trend
    if _trend is None:
        from trend_model import PerCategoryTrendModel
        rt = _load_module(os.path.join(MOD3, "render_trend_demo.py"), "m3_render")
        _trend = (PerCategoryTrendModel.load(TREND_MODEL), rt.render)
    return _trend


def analyze(image_path, module, device):
    """Run the requested module(s) on one photo and return
    (perception_png, harmony_png, trend_png, log_text) — None for figures not requested."""
    if not image_path:
        return None, None, None, "Please upload or paste an outfit photo first."

    want_perception = module in (MODULE_1, ALL)
    want_harmony = module in (MODULE_2, ALL)
    want_trend = module in (MODULE_3, ALL)
    if want_harmony and not os.path.exists(CKPT):
        return None, None, None, f"Trained checkpoint missing: {CKPT}"
    if want_trend and not os.path.exists(TREND_MODEL):
        return None, None, None, (f"Trend model missing: {TREND_MODEL}\n"
                                  "Fit it with: python Module3/fit_per_category.py")

    os.makedirs(DEMOS, exist_ok=True)
    outfit_dir = os.path.join(DEMO_OUT, IMAGE_ID)
    perception_png = os.path.join(DEMOS, "perception_demo.png")
    harmony_png = os.path.join(DEMOS, "harmoniousness_demo.png")
    trend_png = os.path.join(DEMOS, "trend_demo.png")
    dev = None if (not device or device == "auto") else device

    log = []
    t0 = time.time()
    try:
        run_single_image, render = _get_pipeline()

        # Always run Module 1: the perception figure and both scorers read its output.
        log.append(f">> Module 1 — perception on {os.path.basename(image_path)}")
        res = run_single_image(image_path=image_path, image_id=IMAGE_ID,
                               output_root=DEMO_OUT, device=dev)
        log.append(f"   {res['num_garments']} garments, "
                   f"{res['num_embeddings']} embeddings in {res['elapsed_seconds']}s")

        perception_out = None
        if want_perception:
            render(outfit_dir, image_path, perception_png)
            perception_out = perception_png

        harmony_out = None
        if want_harmony:
            log.append(">> Module 2 — GNN Harmoniousness score")
            scorer = _get_scorer()
            score, corr = scorer.score_photo(outfit_dir, image_path, harmony_png)
            log.append(f"   Harmoniousness = {score:.0f}/100"
                       + (f"   (drops to {corr:.0f}/100 if one garment is swapped)"
                          if corr is not None else ""))
            harmony_out = harmony_png

        trend_out = None
        if want_trend:
            log.append(">> Module 3 — per-garment Pinterest trendiness")
            trend_model, render_trend = _get_trend()
            _, result = render_trend(outfit_dir, image_path, trend_png, model=trend_model)
            for g in result["garments"]:
                s = g["trendiness_score"]
                log.append(f"   {g['category']:10s} {g['specific_category']:14s} "
                           + ("no confident match" if s is None else f"{s:5.1f}/100"))
            weak = next((g for g in result["garments"]
                         if g["trendiness_score"] is not None
                         and g["category"] == result["weak_link_category"]
                         and g["trendiness_score"] == result["weak_link_score"]), None)
            if weak:
                log.append(f"   outfit avg {result['outfit_avg_trendiness']:.0f}/100 — weak link: "
                           f"{weak['specific_category']} ({weak['category']}, "
                           f"{weak['trendiness_score']:.0f}/100)")
            trend_out = trend_png

        log.append(f">> done in {time.time() - t0:.1f}s")
        return perception_out, harmony_out, trend_out, "\n".join(log)
    except Exception as e:
        import traceback
        log.append("\n[error] " + str(e))
        log.append(traceback.format_exc())
        return None, None, None, "\n".join(log)


def build_ui():
    with gr.Blocks(title="Outfit Compatibility Demo") as ui:
        gr.Markdown(
            "# Outfit Compatibility Demo\n"
            "Upload or paste a full-body outfit photo, choose which module to run, "
            "then click **Run**.\n\n"
            "- **Module 1 (perception)** — segments the photo into garments and reads "
            "their attributes (colour, pattern, formality, …).\n"
            "- **Module 2 (harmoniousness)** — runs Module 1, then scores the outfit "
            "0–100 with the trained GNN (also shows the drop if one garment is swapped).\n"
            "- **Module 3 (trend)** — runs Module 1, then scores *each garment* 0–100 "
            "against recent Pinterest activity for its category, and flags the "
            "**weak link** (least on-trend piece) — the swap target for Module 4.\n"
            "- **All** — shows all three figures.\n\n"
            "> The **first** run loads the models (~30–60s); every run after that is fast."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(sources=["upload", "clipboard"], type="filepath",
                                 label="Outfit photo (upload or paste)")
                module = gr.Radio([MODULE_1, MODULE_2, MODULE_3, ALL], value=ALL,
                                  label="Which module to run")
                device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                run_btn = gr.Button("Run", variant="primary")
                if os.path.exists(EXAMPLE_PHOTO):
                    gr.Examples(examples=[[EXAMPLE_PHOTO]], inputs=[image],
                                label="Example (samples/image2.jpg)")
            with gr.Column(scale=2):
                perception_img = gr.Image(label="Module 1 — perception", type="filepath")
                harmony_img = gr.Image(label="Module 2 — harmoniousness", type="filepath")
                trend_img = gr.Image(label="Module 3 — per-garment trendiness", type="filepath")
                log = gr.Textbox(label="Log", lines=10)

        run_btn.click(analyze, inputs=[image, module, device],
                      outputs=[perception_img, harmony_img, trend_img, log])
    return ui


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
