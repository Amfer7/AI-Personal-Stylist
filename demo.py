"""
demo.py  —  Gradio GUI for the two completed modules (in-process, models stay warm)
===================================================================================
Upload or paste an outfit photo, choose which module to run, and see the result
figure(s) rendered in the browser:

    Module 1  (perception)     -> segmentation + attributes figure
    Module 2  (harmoniousness) -> end-to-end GNN Harmoniousness score figure
    Both                       -> both figures

Unlike the old CLI, this loads the heavy models (SegFormer + CLIP + the GNN) ONCE
and reuses them for every request — so the *first* run is slow (model load) and
every run after that is fast. It imports the pipeline functions directly instead of
shelling out per click.

Run (from the repo root):
    pip install gradio      # one-time
    python demo.py          # opens the GUI in your browser

Notes:
    - Module 2 needs Module 1's segmentation output, so picking "Module 2" runs
      Module 1 first, then scores.
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
# Module1/run_pipeline.py imports its siblings by bare name; make sure they resolve.
if MOD1 not in sys.path:
    sys.path.insert(0, MOD1)

DATA = os.path.join(HERE, "Fashion144k_v1")
DEMO_OUT = os.path.join(DATA, "demo_out")
DEMOS = os.path.join(HERE, "demos")
CKPT = os.path.join(DATA, "ck_attr_subset_s42", "clip_outfit_gnn_best.pt")
REF_CACHE = os.path.join(DEMOS, "_ref_scores.json")

IMAGE_ID = "demo_live"
EXAMPLE_PHOTO = os.path.join(HERE, "image2.jpg")

MODULE_1 = "Module 1 (perception)"
MODULE_2 = "Module 2 (harmoniousness)"
BOTH = "Both"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- lazy, load-once singletons (heavy models) ----
_pipeline = None   # (run_single_image, render)
_scorer = None     # demo_score.PhotoScorer


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


def analyze(image_path, module, device):
    """Run the requested module(s) on one photo and return
    (perception_png_or_None, harmony_png_or_None, log_text)."""
    if not image_path:
        return None, None, "Please upload or paste an outfit photo first."
    if not os.path.exists(CKPT):
        return None, None, f"Trained checkpoint missing: {CKPT}"

    os.makedirs(DEMOS, exist_ok=True)
    outfit_dir = os.path.join(DEMO_OUT, IMAGE_ID)
    perception_png = os.path.join(DEMOS, "perception_demo.png")
    harmony_png = os.path.join(DEMOS, "harmoniousness_demo.png")

    want_perception = module in (MODULE_1, BOTH)
    want_harmony = module in (MODULE_2, BOTH)
    dev = None if (not device or device == "auto") else device

    log = []
    t0 = time.time()
    try:
        run_single_image, render = _get_pipeline()

        # Always run Module 1: both the perception figure and the scorer read its output.
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

        log.append(f">> done in {time.time() - t0:.1f}s")
        return perception_out, harmony_out, "\n".join(log)
    except Exception as e:
        import traceback
        log.append("\n[error] " + str(e))
        log.append(traceback.format_exc())
        return None, None, "\n".join(log)


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
            "- **Both** — shows both figures.\n\n"
            "> The **first** run loads the models (~30–60s); every run after that is fast."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(sources=["upload", "clipboard"], type="filepath",
                                 label="Outfit photo (upload or paste)")
                module = gr.Radio([MODULE_1, MODULE_2, BOTH], value=BOTH,
                                  label="Which module to run")
                device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                run_btn = gr.Button("Run", variant="primary")
                if os.path.exists(EXAMPLE_PHOTO):
                    gr.Examples(examples=[[EXAMPLE_PHOTO]], inputs=[image],
                                label="Example (image2.jpg)")
            with gr.Column(scale=2):
                perception_img = gr.Image(label="Module 1 — perception", type="filepath")
                harmony_img = gr.Image(label="Module 2 — harmoniousness", type="filepath")
                log = gr.Textbox(label="Log", lines=8)

        run_btn.click(analyze, inputs=[image, module, device],
                      outputs=[perception_img, harmony_img, log])
    return ui


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
