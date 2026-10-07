# Module 3: Trend (per-garment, from Pinterest)

**In one line:** scores **each garment** 0–100 for how present its style is in recent Pinterest
activity.

```
Pinterest photos ──► garments ──► style clusters per category ──► activity over time ──► trend score per cluster
your garment ──► nearest style cluster ──► that cluster's trend score
```

**In plain terms:** we collected ~61k fashion photos from Pinterest with their posting dates, cut out
every garment (using Module 1's code), and grouped similar-looking garments into **style clusters**,
separately for tops, bottoms, dresses and accessories. Each cluster gets a score from how much it shows
up recently. To score one of your garments, we find the cluster it looks most like.

**Important:** the score is **descriptive** ("how present is this style now"), not a forecast. A
backtest showed recent growth does *not* predict future growth (see Results).

---

## Technical summary

Scores **each garment** in an outfit (top / bottom / dress / accessory) and flags the least on-trend
piece. One independent **K-Means** trend model is fit per garment category on 512-D CLIP embeddings,
so "which tops are trending" and "which bottoms are trending" are answered separately. Each cluster's
monthly counts give a **growth** and a **recency** signal, which are converted to percentile ranks and
combined into a 0–100 score.

Segmentation and embedding reuse Module 1's code verbatim (same SegFormer, same
512-D L2-normalized CLIP ViT-B/32), so Pinterest garments and user-photo
garments live in the same vector space.

## Files

| File | What it is |
|---|---|
| `trend_model.py` | `TrendModel` (cluster → monthly curve → growth + recency → 0–100 score) and `PerCategoryTrendModel` (one `TrendModel` per category). Also holds the shared k rule (`k_for_category`, `N_CLUSTERS_OVERRIDES`) and `fragment_mask` |
| `pinterest_garment_segmentation.py` | Batch-segments the Pinterest scrape into garments and embeds each crop. Resumable, append-only checkpoints, circuit breaker. `--mode lean` (default) / `--mode full` |
| `fit_per_category.py` | Fits the `PerCategoryTrendModel` from the manifest + embeddings → `per_category_trend_model.pkl` |
| `cluster_stability_test.py` | 80/20 resampling stability per category (Adjusted Rand Index) |
| `temporal_holdout_validation.py` | Backtests whether the growth signal predicts *forward* growth. Whole-photo (`--metadata_csv`) or per-garment (`--manifest_csv`) mode |
| `k_sweep.py` | Sweeps k for one category (ARI + silhouette) — how the dress k was chosen |
| `render_trend_demo.py` | Scores a Module 1 output folder and renders the slide figure (`demos/trend_demo.png`) |
| `gradio_demo_per_garment.py` | Standalone per-garment UI (segments → embeds → scores → weak link) |
| `test_real_photo_per_garment.py` | CLI: score one photo |
| `gradio_demo.py`, `synthetic_demo.py` | Original whole-photo demo / synthetic-data smoke test |
| `fashion_segmenter.py`, `attribute_analyzer.py`, `clip_extract_embeddings.py` | Vendored from `Module1/` — **must stay identical**, enforced by `test_module1_sync.py` |
| `test_module1_sync.py`, `test_embedding_parity.py` | Guards: vendored-file drift; lean batched embedding == `embed_image` |

## Data

Everything is local, in-repo (gitignored):

- `PinterestData/pinterest_dataset/` — 61,118 scraped pins (`metadata.csv` + `images/`), dated 2011-10 → **2026-09-11** (scrape end).
- `PinterestData/garment_data/` — `garment_manifest.csv` + `garment_embeddings.npy`: **143,612 garments** from 54,876 photos
  (top 48,871 · bottom 42,728 · accessory 32,175 · dress 19,838; shoes/bags/hats/scarves/belts/sunglasses are `accessory`).
  `garment_embeddings.f32` is the raw resume buffer — redundant once the `.npy` exists.

**Lean vs full mode.** Lean mode (used for the run above) skips the two CLIP
zero-shot passes the trend model never reads, so `specific_category` is the
coarse SegFormer label (`upper_clothes`, `pants`, …) and `confidence` is always
1.0. Use `--mode full` if fine-grained subcategories are ever needed.
`test_embedding_parity.py` verifies the lean embeddings are identical.

## Pipeline

Run from `Module3/`:

```bash
python test_module1_sync.py && python test_embedding_parity.py      # guards first

python pinterest_garment_segmentation.py \
    --metadata_csv ../PinterestData/pinterest_dataset/metadata.csv \
    --output_dir   ../PinterestData/garment_data --limit 1000      # pilot; drop --limit for the full run

python fit_per_category.py                                           # -> per_category_trend_model.pkl
python cluster_stability_test.py \
    --manifest_csv ../PinterestData/garment_data/garment_manifest.csv \
    --embeddings   ../PinterestData/garment_data/garment_embeddings.npy
python temporal_holdout_validation.py \
    --manifest_csv ../PinterestData/garment_data/garment_manifest.csv \
    --embeddings   ../PinterestData/garment_data/garment_embeddings.npy \
    --n_clusters 15 --end_month 2026-08

python gradio_demo_per_garment.py                                    # standalone UI
```

The combined M1 + M2 + M3 demo is the repo-root `demo.py` (pick **Module 3** or **All**).

`fit_per_category.py` defaults to `--end_month 2026-08`. The scrape ended on
2026-09-11, so September is a partial month that would otherwise sit inside the
growth/recency windows. That leaves 126,898 garments in the fit.

## Results

**Cluster stability** (80/20 resampling, ARI, 5 repeats — all "strong"):

| category | k | ARI |
|---|---|---|
| accessory | 15 | 0.885 |
| bottom | 15 | 0.881 |
| dress | **8** | **0.864** (was 0.58 at k=15) |
| top | 15 | 0.759 |

*Dress fix.* `k_sweep.py` showed the instability came from k, not the data
(ARI by k: 4→0.95, 8→0.94, 12→0.89, 15→0.58), so dresses use k=8
(`N_CLUSTERS_OVERRIDES`). ~Half of dress detections co-occur with a top in the
same photo, and in 3,992 the dress mask is the *smaller* one, which is likely a
SegFormer fragment (e.g. a long top's hem). `fit_per_category.py --drop_fragments` removes
them; it's off by default because it didn't change stability at k=8 and a small
"dress" under a large cardigan can be genuine. The vendored segmenter is not
modified.

**Temporal holdout — does growth predict the future?** No. Fit on data up to
2026-05, predict growth over the next 3 months (Jun–Aug 2026), compare with what happened:

Spearman ρ, predicted vs actual growth, per category (15 clusters; dress 8).
"Pooled" ranks within each category, then correlates all 53 clusters:

| window | accessory | bottom | dress | top | pooled ρ (n=53) |
|---|---|---|---|---|---|
| 1 mo | −0.45 | −0.59 | −0.36 | −0.39 | **−0.45** (p<0.001) |
| 3 mo (model default) | −0.31 | 0.25 | −0.86 | 0.20 | −0.10 (p=0.48) |
| 6 mo | −0.10 | −0.34 | −0.55 | −0.51 | **−0.35** (p=0.01) |

(For 1- and 6-month windows the holdout is the last 1 / 6 months, so the train cutoff moves accordingly.)

The growth signal has no forward predictive power at the default 3-month window.
At 1 and 6 months it is *negatively* predictive: clusters that just surged tend to
fall back (mean reversion). Raw and share-normalized growth give identical
rankings, because normalizing divides every cluster by the same monthly total.

**What this means for the score.** Trendiness is a *descriptive* score:
"how present/rising is this style in recent Pinterest activity". It is not a
forecast. The demo bands ("Rising", "Frequently seen now") are worded to
match. Pinterest's results also skew heavily recent (~500 garments/month in
2024 vs ~27k in Aug 2026), so absolute volumes mean little and only *relative*
comparisons across clusters are used (scores are percentile ranks).

## Tunable config (`trend_model.py`)

`DEFAULT_N_CLUSTERS_PER_CATEGORY=15`, `N_CLUSTERS_OVERRIDES={"dress": 8}`,
`GROWTH_WINDOW_MONTHS=3`, `RECENCY_HALF_LIFE_MONTHS=4`, `GROWTH_WEIGHT=0.6` /
`RECENCY_WEIGHT=0.4`, `MIN_CLUSTER_IMAGES=15`, `MATCH_CONFIDENCE_THRESHOLD=0.20`.
Given the backtest, the growth/recency weighting is being re-decided: Module 4's design (decision
D11, [`docs/module4-design.md`](../docs/module4-design.md)) re-tests it against "is this style still
widely seen months later?" after M3 is refit on full-resolution Pinterest photos.

## Why not "train in epochs like the GNN"

K-Means has no loss curve to train epoch-over-epoch; it's a one-shot fit. The
correctly-scoped validations are **stability under resampling** (does the
clustering reflect real structure?) and the **temporal backtest** (does the
growth signal predict forward?). These answer two different questions.

## Integration point for Module 4

`PerCategoryTrendModel.score_outfit(garments)` takes
`[{"embedding": (512,), "category": broad_category, ...}]` and returns
per-garment `trendiness_score`, `outfit_avg_trendiness`, and
`weak_link_category` / `weak_link_score`. `render_trend_demo.score_outfit_dir()` shows how
to feed it straight from a Module 1 output folder.

Module 4 uses the **per-garment** scores and cluster ids, not the trend weak link: it decides which
piece to swap by trying replacements and combining harmony with trend (see
[`docs/module4-design.md`](../docs/module4-design.md)).
