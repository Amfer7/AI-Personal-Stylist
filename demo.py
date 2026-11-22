"""
demo.py  —  Gradio GUI for all four modules (in-process, models stay warm)
==========================================================================
Upload or paste an outfit photo, choose which module to run, and see the results in the browser:

    Module 1  (perception)     -> segmentation + attributes figure
    Module 2  (harmoniousness) -> end-to-end GNN Harmoniousness score figure
    Module 3  (trend)          -> per-garment Pinterest trendiness figure
    Module 4  (recommend)      -> which piece to swap and what for, with a live trend-weight slider
    All                        -> everything

The heavy models (SegFormer + CLIP, the GNN, the trend model, Module 4's candidate pool) load ONCE
and are reused for every request, so the *first* run is slow and every run after that is fast.
Modules 2 and 4 share one GNN and one harmony reference (all real test outfits of both catalogues),
so they report the same harmony % for your outfit.

Run (from the repo root):
    pip install gradio      # one-time
    python demo.py          # opens the GUI in your browser
    python demo.py --vram-gb 4    # lower GPU-memory cap (default 6, from Module4/m4_settings.json)

Notes:
    - Modules 2-4 need Module 1's segmentation output, so every run starts with Module 1.
    - Module 4 needs its cache: python Module4/build_cache.py
    - Assumes the checkpoint, feature stores and trend model named in Module4/m4_settings.json.
"""

import argparse
import os
import sys
import time
import importlib.util

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
MOD1 = os.path.join(HERE, "Module1")
MOD2 = os.path.join(HERE, "Module2")
MOD3 = os.path.join(HERE, "Module3")
MOD4 = os.path.join(HERE, "Module4")
# Module1/run_pipeline.py imports its siblings by bare name; make sure they resolve.
if MOD1 not in sys.path:
    sys.path.insert(0, MOD1)
# Module3 and Module4 are appended (not prepended) so Module1's originals win for the vendored
# names they share; the trend pickle needs `trend_model` importable by bare name.
for _p in (MOD3, MOD4):
    if _p not in sys.path:
        sys.path.append(_p)

from m4_settings import M4Settings  # noqa: E402  (light: reads JSON only)

SETTINGS = M4Settings.load()
DATA = os.path.join(HERE, "Fashion144k_v1")
DEMO_OUT = os.path.join(DATA, "demo_out")
DEMOS = os.path.join(HERE, "demos")
CKPT = SETTINGS.abs(SETTINGS.checkpoint)
TREND_MODEL = SETTINGS.abs(SETTINGS.trend_model)      # the same model Module 4 ranks with
BOARD_PNG = os.path.join(DEMOS, "recommendation_board.png")

IMAGE_ID = "demo_live"
EXAMPLE_PHOTO = os.path.join(HERE, "samples", "image2.jpg")

MODULE_1 = "Module 1 (perception)"
MODULE_2 = "Module 2 (harmoniousness)"
MODULE_3 = "Module 3 (trend)"
MODULE_4 = "Module 4 (recommend)"
ALL = "All"
POOLS = ["demo", "eval"]
RULES = ["max_gain", "topk_mean_gain", "slot_percentile"]
# pool, same_type, no_trend_drop, rule, w_t (the calibrated settings)
DEFAULT_OPTIONS = ("demo", False, False, SETTINGS.rule, SETTINGS.w_t)

M4_OUTPUT_NAMES = ["m4_verdict", "m4_gallery", "m4_also", "m4_table", "m4_choice", "m4_board",
                   "m4_board_text", "m4_state"]
OUTPUT_NAMES = ["perception", "harmony", "trend", "log"] + M4_OUTPUT_NAMES


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- lazy, load-once singletons (heavy models) ----
_pipeline = None      # (run_single_image, render)
_scorer = None        # demo_score.PhotoScorer
_trend = None         # (trend model, render_trend_demo.render)
_recommender = None   # Module 4 Recommender (GNN + trend model + candidate pool)


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


def _get_recommender(device=None):
    """Module 4: GNN, trend model and the candidate-pool cache, loaded once."""
    global _recommender
    if _recommender is None:
        _get_pipeline()   # import Module1's modules first: m2_bridge puts Module3 at the front of sys.path
        from recommender import Recommender
        _recommender = Recommender.from_settings(SETTINGS, device)
    return _recommender


def _reference_path():
    from candidate_pool import pool_dir
    return os.path.join(pool_dir(SETTINGS), "reference.json")


def _get_scorer(device=None):
    """Module 2: PhotoScorer, ranking against Module 4's harmony reference and reusing Module 4's
    GNN (loading Module 4 first), so there is only ever one GNN copy."""
    global _scorer
    if _scorer is None:
        from harmony_reference import HarmonyReference
        ref = HarmonyReference.load(_reference_path())
        model = _get_recommender(device).gnn.model
        ds_mod = _load_module(os.path.join(MOD2, "demo_score.py"), "m2_demo_score")
        _scorer = ds_mod.PhotoScorer(checkpoint=CKPT, model=model, reference=ref.logits_sorted)
    return _scorer


def _get_trend(device=None):
    """Module 3: the trend model Module 4 uses (loading Module 4 first, so it is shared) + its renderer."""
    global _trend
    if _trend is None:
        rt = _load_module(os.path.join(MOD3, "render_trend_demo.py"), "m3_render")
        if os.path.exists(_reference_path()):
            model = _get_recommender(device).trend_model
        else:        # Module 4 cache not built: Module 3 still works on its own
            from trend_model import PerCategoryTrendModel
            model = PerCategoryTrendModel.load(TREND_MODEL)
        _trend = (model, rt.render)
    return _trend


# ---- Module 2 and Module 4 steps (also called by the tests) ----
def run_module2(outfit_dir, image_path, out_png, device=None):
    """Harmoniousness on Module 4's 98,304-px area frame, so it matches Module 4's baseline."""
    from outfit_state import REFERENCE_FRAME_PX, photo_pixels
    px = photo_pixels(outfit_dir)
    return _get_scorer(device).score_photo(outfit_dir, image_path, out_png,
                                     area_scale=REFERENCE_FRAME_PX / px if px else 1.0)


def run_module4(outfit_dir, pool_mode, same_type, no_trend_drop, rule, w_t, device=None):
    """The only GNN stage: score every allowed replacement once, then decide."""
    from outfit_state import load_outfit_dir
    rec = _get_recommender(device)
    state = load_outfit_dir(outfit_dir, rec.gnn, rec.trend_model)
    table = rec.score(state, mode=pool_mode, same_type=bool(same_type), trend_floor=bool(no_trend_drop))
    return {"table": table, "rec": rec.decide(table, w_t=w_t, w_h=1.0 - w_t, rule=rule)}


def _m4_empty(message=""):
    return ("", None, "", None, gr.Dropdown(choices=[], value=None), None, message, None)


def _m4_outputs(st):
    from demo_panel import (also_markdown, choice_labels, gallery_items, per_piece_rows,
                            swap_board, verdict_markdown)
    rec, r = _get_recommender(), st["rec"]
    labels = choice_labels(r)
    if labels:
        board, text = swap_board(r, st["table"].state, rec.pool, rec.settings, 0, BOARD_PNG)
    else:
        board, text = None, "Nothing to try: the outfit is already strong."
    return (verdict_markdown(r), gallery_items(r, rec.settings), also_markdown(r), per_piece_rows(r),
            gr.Dropdown(choices=labels, value=labels[0] if labels else None), board, text, st)


def rerank(st, w_t, rule):
    """Slider / rule change: re-decide from the cached ΔH and ΔT. No GNN call."""
    if not st:
        return _m4_empty("Run Module 4 first.")
    rec = _get_recommender()
    return _m4_outputs({"table": st["table"],
                        "rec": rec.decide(st["table"], w_t=w_t, w_h=1.0 - w_t, rule=rule)})


def try_swap(st, label):
    """"Try this swap": the outfit board with the chosen suggestion swapped in."""
    from demo_panel import choice_labels, swap_board
    if not st or not label:
        return None, ""
    rec, r = _get_recommender(), st["rec"]
    labels = choice_labels(r)
    if label not in labels:
        return None, "That option is no longer listed: press Run again."
    return swap_board(r, st["table"].state, rec.pool, rec.settings, labels.index(label), BOARD_PNG)


def analyze(image_path, module, device, pool_mode, same_type, no_trend_drop, rule, w_t):
    """Run the requested module(s) on one photo; returns values in the order of OUTPUT_NAMES."""
    if not image_path:
        return (None, None, None, "Please upload or paste an outfit photo first.") + _m4_empty()

    want_perception = module in (MODULE_1, ALL)
    want_harmony = module in (MODULE_2, ALL)
    want_trend = module in (MODULE_3, ALL)
    want_m4 = module in (MODULE_4, ALL)
    if (want_harmony or want_m4) and not os.path.exists(CKPT):
        return (None, None, None, f"Trained checkpoint missing: {CKPT}") + _m4_empty()
    if (want_trend or want_m4) and not os.path.exists(TREND_MODEL):
        return (None, None, None, f"Trend model missing: {TREND_MODEL}\n"
                                  "Fit it with: python Module3/fit_per_category.py") + _m4_empty()
    if (want_harmony or want_m4) and not os.path.exists(_reference_path()):
        return (None, None, None, "Module 4 cache missing: run python Module4/build_cache.py") + _m4_empty()

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
        if want_m4:
            _get_recommender(dev)          # load first, so Module 2 and 3 reuse its GNN + trend model

        # Always run Module 1: the perception figure and every scorer read its output.
        log.append(f">> Module 1 — perception on {os.path.basename(image_path)}")
        res = run_single_image(image_path=image_path, image_id=IMAGE_ID,
                               output_root=DEMO_OUT, device=dev)
        if res.get("status") == "animal_no_person":     # stop: the outfit folder still holds the last photo
            log.append(f"   {res['message']}")
            return (None, None, None, "\n".join(log)) + _m4_empty(res["message"])
        log.append(f"   {res['num_garments']} garments, "
                   f"{res['num_embeddings']} embeddings in {res['elapsed_seconds']}s")
        if not res["num_garments"]:
            msg = "No garments found in this photo. Try a clearer, full-body photo of a person."
            log.append(f"   {msg}")
            return (None, None, None, "\n".join(log)) + _m4_empty(msg)

        perception_out = None
        if want_perception:
            render(outfit_dir, image_path, perception_png)
            perception_out = perception_png

        harmony_out = None
        if want_harmony:
            log.append(">> Module 2 — GNN Harmoniousness score")
            score, corr = run_module2(outfit_dir, image_path, harmony_png, dev)
            log.append(f"   Harmoniousness = {score:.0f}/100"
                       + (f"   (drops to {corr:.0f}/100 if one garment is swapped)"
                          if corr is not None else ""))
            harmony_out = harmony_png

        trend_out = None
        if want_trend:
            log.append(">> Module 3 — per-garment Pinterest trendiness")
            trend_model, render_trend = _get_trend(dev)
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
                log.append(f"   outfit avg {result['outfit_avg_trendiness']:.0f}/100 — least on-trend: "
                           f"{weak['specific_category']} ({weak['category']}, "
                           f"{weak['trendiness_score']:.0f}/100)")
            trend_out = trend_png

        m4 = _m4_empty()
        if want_m4:
            log.append(">> Module 4 — swap recommendation")
            t4 = time.time()
            st = run_module4(outfit_dir, pool_mode, same_type, no_trend_drop, rule, w_t, dev)
            r = st["rec"]
            n = sum(len(p.rows) for p in st["table"].pieces)
            log.append(f"   verdict: {r.verdict}" + (f" ({r.swap_piece['category']})" if r.swap_piece else "")
                       + f" · {n} what-if outfits scored in {time.time() - t4:.2f}s")
            m4 = _m4_outputs(st)

        log.append(f">> done in {time.time() - t0:.1f}s")
        return (perception_out, harmony_out, trend_out, "\n".join(log)) + tuple(m4)
    except Exception as e:
        import traceback
        log.append("\n[error] " + str(e))
        log.append(traceback.format_exc())
        return (None, None, None, "\n".join(log)) + _m4_empty()


def build_ui():
    from demo_panel import PER_PIECE_HEADERS
    with gr.Blocks(title="Outfit Compatibility Demo") as ui:
        gr.Markdown(
            "# Outfit Compatibility Demo\n"
            "Upload or paste a full-body outfit photo, choose which module to run, "
            "then click **Run**.\n\n"
            "- **Module 1 (perception)** — segments the photo into garments and reads "
            "their attributes (colour, pattern, formality, …).\n"
            "- **Module 2 (harmoniousness)** — scores the outfit 0–100 with the trained GNN "
            "(also shows the drop if one garment is swapped).\n"
            "- **Module 3 (trend)** — scores *each garment* 0–100 for current popularity on "
            "Pinterest in its category, and marks the least on-trend piece.\n"
            "- **Module 4 (recommend)** — tries every allowed replacement for every piece and says "
            "which one piece to swap and what for, or that the outfit is already strong.\n"
            "- **All** — everything.\n\n"
            "> The **first** run loads the models (~30–60s); every run after that is fast."
        )
        m4_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(sources=["upload", "clipboard"], type="filepath",
                                 label="Outfit photo (upload or paste)")
                module = gr.Radio([MODULE_1, MODULE_2, MODULE_3, MODULE_4, ALL], value=ALL,
                                  label="Which module to run")
                device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device (applies on the first run)")
                w_t = gr.Slider(0.0, 1.0, value=SETTINGS.w_t, step=0.05,
                                label="Trend weight (Module 4): 0 = harmony only, 1 = trend only")
                with gr.Accordion("Module 4 advanced options (pool, same type and trend floor apply on the next Run)", open=False):
                    pool = gr.Radio(POOLS, value="demo",
                                    label="Candidate pool (demo = all garments, eval = test split only)")
                    same_type = gr.Checkbox(False, label="Same type only (strict like-for-like)")
                    no_trend_drop = gr.Checkbox(False, label="Never suggest a less trendy piece (trend floor)")
                    rule = gr.Dropdown(RULES, value=SETTINGS.rule, label="Selection rule")
                run_btn = gr.Button("Run", variant="primary")
                if os.path.exists(EXAMPLE_PHOTO):
                    gr.Examples(examples=[[EXAMPLE_PHOTO]], inputs=[image],
                                label="Example (samples/image2.jpg)")
            with gr.Column(scale=2):
                perception_img = gr.Image(label="Module 1 — perception", type="filepath")
                harmony_img = gr.Image(label="Module 2 — harmoniousness", type="filepath")
                trend_img = gr.Image(label="Module 3 — per-garment trendiness", type="filepath")
                gr.Markdown("### Module 4 — recommendation")
                verdict = gr.Markdown()
                gallery = gr.Gallery(label="Suggestions", columns=3, height=320)
                also = gr.Markdown()
                with gr.Row():
                    choice = gr.Dropdown(choices=[], label="Try this swap")
                board = gr.Image(label="Your outfit with the swap", type="filepath")
                board_text = gr.Markdown()
                table = gr.Dataframe(headers=PER_PIECE_HEADERS, label="Per piece", interactive=False)
                log = gr.Textbox(label="Log", lines=10)

        m4_outputs = [verdict, gallery, also, table, choice, board, board_text, m4_state]
        run_btn.click(analyze,
                      inputs=[image, module, device, pool, same_type, no_trend_drop, rule, w_t],
                      outputs=[perception_img, harmony_img, trend_img, log] + m4_outputs)
        w_t.release(rerank, inputs=[m4_state, w_t, rule], outputs=m4_outputs)
        rule.change(rerank, inputs=[m4_state, w_t, rule], outputs=m4_outputs)
        choice.change(try_swap, inputs=[m4_state, choice], outputs=[board, board_text])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Outfit compatibility demo (Modules 1-4)")
    ap.add_argument("--vram-gb", type=float, default=None, dest="vram_gb",
                    help=f"GPU memory cap for Module 4 (default {SETTINGS.vram_gb} from m4_settings.json)")
    args = ap.parse_args()
    if args.vram_gb is not None:
        SETTINGS = SETTINGS.with_(vram_gb=args.vram_gb)
    build_ui().launch(inbrowser=True)
