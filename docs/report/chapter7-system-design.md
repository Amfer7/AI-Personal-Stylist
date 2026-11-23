# Chapter 7 – System Design

Chapter 5 described the system's architecture and Chapter 6 described how each module works inside.
This chapter covers how the complete application is put together: how the user-facing application
coordinates the four modules, what each module hands to the next, how information is represented
and stored, and how the user interface connects to the processing code. Algorithms already covered
in Chapter 6 (segmentation, GNN message passing, BPR training, trend scoring, swap selection) are
referred to by name only.

---

## 7.1 Overall System Design

### 7.1.1 Organisation of the application

The system runs as a single local Python process. One file, `demo.py`, is both the entry point and
the orchestration layer: it builds the Gradio user interface and routes each request to the
modules. Every module is called **in-process**, as an ordinary Python function or object. There are
no network services, message queues or separate worker processes between the modules.

The system is split into two phases that run at different times:

| Phase | When it runs | What it does | Entry points |
|---|---|---|---|
| **Offline build** | Once, before the application starts (repeated only when the data or models change) | Segments the three garment datasets with Module 1, builds the feature stores, trains the GNN, fits the trend model, aligns the H&M catalogue and builds the Module 4 candidate-pool cache | `Module2/src/01_segment_batch.py`, `Module3/scripts/pinterest_garment_segmentation.py`, `Module4/src/hm_catalogue.py`, `Module2/scripts/build_feature_store.py`, `Module2/src/03_train_gnn.py`, `Module3/scripts/fit_per_category.py`, `Module4/scripts/build_cache.py` |
| **Online inference** | For each photo a user submits | Segments the photo, scores it and produces a recommendation, using only the assets the offline phase built | `demo.py` (GUI); `Module1/src/run_pipeline.py` and `Module4/src/run_recommend.py` (command line) |

No model is trained and no feature store is built while a user is waiting. At request time the
application only reads assets the offline phase produced.

### 7.1.2 Central orchestration in `demo.py`

`demo.py` has three layers:

1. **Load-once model holders.** Four module-level variables hold the heavy objects. Each is filled
   the first time it is needed and reused for every later request:
   - `_get_pipeline()` loads Module 1 (`run_single_image` and the perception renderer). The
     SegFormer, CLIP and YOLO models cache themselves in module globals inside Module 1
     (`fashion_segmenter.py`, `clip_extract_embeddings.py`, `subject_check.py`).
   - `_get_recommender()` builds the Module 4 `Recommender` with `Recommender.from_settings()`.
     That call loads the GNN, the trend model, the candidate pool, the harmony reference and the
     co-occurrence filter.
   - `_get_scorer()` builds the Module 2 `PhotoScorer`, but passes it **Module 4's GNN instance and
     harmony reference** (`model=..., reference=...`), so only one copy of the GNN is ever in
     memory.
   - `_get_trend()` reuses **Module 4's trend model** when the Module 4 cache exists. Without the
     cache it loads the trend pickle on its own, so Module 3 still works without Module 4.
2. **Request handler.** `analyze()` receives the photo and the user's options, decides which
   modules to run, checks that the required assets exist, calls the modules in order and returns
   one value per output widget.
3. **Interactive callbacks.** `rerank()` and `try_swap()` let the user change the trend weight or
   selection rule, or preview a swap, without running the pipeline again (Section 7.5).

The first request takes roughly 30–60 seconds while the models load (as stated in the UI text).
Later requests reuse the loaded models.

### 7.1.3 Execution model: sequential with conditional branches

Execution is **neither fully sequential nor fully independent**. Module 1 always runs first.
Modules 2, 3 and 4 then run only if the user selected them, in a fixed order:

```
analyze(photo, module, options)
 ├─ validate inputs and required assets
 ├─ if Module 4 or All: load the Recommender first (so M2 and M3 share its GNN and trend model)
 ├─ Module 1 run_single_image()        ← ALWAYS (every other module reads its output folder)
 │    ├─ status "animal_no_person" → stop with a message
 │    └─ 0 garments found          → stop with a message
 ├─ if Module 1 or All: render perception figure
 ├─ if Module 2 or All: run_module2()  → harmony figure + score
 ├─ if Module 3 or All: render_trend() → trend figure + per-garment scores
 └─ if Module 4 or All: run_module4()  → SwapTable + Recommendation → M4 widgets
```

Module 1 always runs because Modules 2–4 do not take the image itself as input. They read the
garment folder Module 1 writes. Modules 2, 3 and 4 do not depend on one another's results within a
request. Module 4 calls Module 2's GNN and Module 3's trend model directly (Section 7.2) and does not
use the Module 2 or Module 3 figures. Choosing **All** runs Modules 2, 3 and 4 one after another on
the same Module 1 output.

### 7.1.4 Dependency handling

`analyze()` checks for the required assets before any processing starts:

| Selected module | Assets checked | Message if missing |
|---|---|---|
| Module 2 or Module 4 | GNN checkpoint (`settings.checkpoint`) | "Trained checkpoint missing: …" |
| Module 3 or Module 4 | Trend model pickle (`settings.trend_model`) | "Trend model missing … Fit it with: python Module3/scripts/fit_per_category.py" |
| Module 2 or Module 4 | Module 4 cache (`Module4/cache/pool/reference.json`) | "Module 4 cache missing: run python Module4/scripts/build_cache.py" |

Module 2 depends on the Module 4 cache because the application ranks Module 2's harmony score
against the same reference distribution Module 4 uses. Both modules therefore report the same
harmony percentage for a given outfit. As a result, **Module 2 cannot run in the GUI unless the
Module 4 cache has been built**, while Module 3 can.

All file paths and settings come from one configuration file, `Module4/config/m4_settings.json`, read
through the `M4Settings` dataclass (`Module4/src/m4_settings.py`). Paths are stored relative to the
repository and resolved with `M4Settings.abs()`, so no machine-specific path is stored.

### 7.1.5 GUI and command-line paths

The same module functions are reachable without the GUI:

| Path | Entry point | What it does |
|---|---|---|
| GUI | `python demo.py [--vram-gb N]` | All modules, interactive; models stay loaded between requests |
| Module 1 CLI | `python Module1/src/run_pipeline.py --image <photo>` or `--input_dir <folder>` | Segmentation and embeddings for one image or a folder (batch) |
| Module 4 CLI | `python Module4/src/run_recommend.py --photo <photo>` or `--outfit_dir <M1 folder>` | Runs Module 1 if given a photo, then prints the recommendation as text; `--json` saves the `Recommendation` |
| Module 2 CLI | `python Module2/src/demo_score.py …` | Real vs corrupted harmony demo over test outfits |
| Module 3 CLI | `python Module3/src/render_trend_demo.py …` | Per-garment trend figure for one outfit folder |

The GUI and the Module 4 CLI call the same code (`Recommender.score()` and `Recommender.decide()`).
They differ only in where Module 1 writes its output: `Fashion144k_v1/demo_out/demo_live/` for the
GUI and `Module4/cache/runs/<photo name>/` for the CLI. Module 3 also contains two older
stand-alone Gradio apps (`Module3/scripts/gradio_demo.py`, `Module3/scripts/gradio_demo_per_garment.py`) that
`demo.py` does not use.

---

## 7.2 Module-Level Design

### 7.2.1 Role of each module in the complete system

| Module | Role in the system | Receives | Returns |
|---|---|---|---|
| **M1 Perception** | Turns a photo into a set of described garments; also run offline to build the garment catalogues | Image file path, image id, output root, device | A result dict (`status`, `num_garments`, `num_embeddings`, `output_dir`, `garments`, `elapsed_seconds`) **plus an outfit folder on disk** |
| **M2 Harmony** | Scores how well an outfit's pieces go together | An M1 outfit folder (GUI); projected node tensors (inside M4) | A harmony percentile (0–100) and the score after one corrupting swap; raw GNN logits inside M4 |
| **M3 Trend** | Scores how popular each garment's style currently is | A garment's 512-D CLIP embedding and broad category | `trendiness_score` (0–100 or `None`), `cluster_id`; per-outfit average and least on-trend piece |
| **M4 Recommendation** | Decides which one piece to swap and what to replace it with | An M1 outfit folder, pool mode, swap options, trend weight, selection rule | A `SwapTable` (every scored candidate) and a `Recommendation` (verdict, suggestions, per-piece table) |

### 7.2.2 The shared garment representation

All four modules describe a garment the same way, as defined by Module 1:

- a **transparent RGBA crop** (`<type>_<n>.png`);
- a **512-D CLIP ViT-B/32 embedding** (`<type>_<n>.npy`), produced by Module 1's
  `clip_extract_embeddings.py`;
- a **metadata entry** in `metadata.json`: file name, `category` (specific type),
  `broad_category`, `raw_category`, confidence, `pixel_count`, bounding box and an `attributes`
  block (colour with hue, silhouette, pattern, fabric, fit, formality and `formality_score`).

Module 2's `build_node_from_meta()` (`Module2/src/02_outfit_dataset.py`) turns one metadata entry into
a GNN node: the embedding, a 32-bin colour histogram computed from the crop, encoded attributes,
and the three edge scalars (hue, formality, area). The **feature stores** hold the same node
contents precomputed for every catalogue garment, as one 570-D row
`[512 CLIP | 32 colour | 26 attributes]` plus a scalar row `[hue, formality, area]`. Because a user
photo and a catalogue garment go through the same code, a garment from a user photo can be
compared directly with a catalogue garment.

Module 3 keeps its own copies of three Module 1 files (`fashion_segmenter.py`,
`attribute_analyzer.py`, `clip_extract_embeddings.py`), so it can segment the Pinterest dataset on
its own. `Module3/tests/test_module1_sync.py` checks that the copies stay identical to the originals, so
the trend model is fitted on the same embeddings the application produces.

### 7.2.3 Interfaces between modules

| Interface | Mechanism | Data exchanged | Code |
|---|---|---|---|
| **M1 → M2** | **File-based.** M2 reads the folder M1 wrote | `metadata.json`, `<type>_n.npy`, `<type>_n.png`; the photo size comes from `<id>_vis.jpg` for area normalisation | `demo.run_module2()` → `PhotoScorer.score_photo()` → `od.build_node_from_meta()` |
| **M1 → M3** | **File-based** | `metadata.json` (file, broad and specific category) and each `.npy` embedding | `render_trend_demo.score_outfit_dir()` → `PerCategoryTrendModel.score_outfit()` |
| **M1 → M4** | **File-based**, then converted to in-memory objects | The same folder, loaded into an immutable `OutfitState` of `Piece` objects | `outfit_state.load_outfit_dir()` |
| **M2 → M4** | **Direct Python import** through one adapter module | The `CLIPOutfitGNN` model and its weights, the feature-column mask, the edge-weight table and the node-feature builder | `Module4/src/m2_bridge.py` (`load_gnn`, `node_features`, `edge_weight_table`, `FEATURE_COLS`) |
| **M3 → M4** | **Direct Python import** (pickled object) | The `PerCategoryTrendModel` object: centroids per category, cluster trend scores, match threshold | `m2_bridge.load_trend_model()`; vectorised scoring in `candidate_pool.trend_scores()` |
| **M4 → GUI** | **Python objects** | `{"table": SwapTable, "rec": Recommendation}`, converted to widget values by `Module4/src/demo_panel.py` | `demo.run_module4()`, `demo._m4_outputs()` |

**How Module 4 reaches Module 2.** Module 2's training files begin with digits
(`02_outfit_dataset.py`, `03_train_gnn.py`), so they cannot be imported by name.
`Module4/src/m2_bridge.py` loads them by file path, once, and is the only way into Module 2 from
Module 4. It exposes the trained network (`load_gnn`), the 555 input columns the shipped checkpoint
reads (`FEATURE_COLS`: 512 CLIP + 32 colour + 11 selected attributes) and the category edge-weight
table. Module 4 does not run Module 2's Python graph builder for each candidate. Instead,
`Module4/src/dense_gnn.py` (`DenseGNN`) recomputes the same logits with batched tensor operations,
using the loaded model's own layers and weights. `test_dense_gnn.py` checks that it gives the same
result as Module 2's original path.

**How Module 4 reaches Module 3.** The trend model is a pickled `PerCategoryTrendModel`.
Unpickling it requires the class `trend_model` to be importable by name, so `m2_bridge.py` adds
`Module3/` to `sys.path`. Module 4 uses the model in two ways: per piece of the user's outfit
(`score_embedding()` in `load_outfit_dir`), and once for the whole catalogue at cache-build time
(`trend_scores()`, a vectorised version of the same nearest-centroid lookup).

### 7.2.4 Shared instances

| Shared item | Shared by | How |
|---|---|---|
| GNN model (one instance in memory) | M2 and M4 | `_get_scorer()` passes `_get_recommender().gnn.model` to `PhotoScorer` |
| Harmony reference (logits of 56,273 real test outfits) | M2 and M4 | `HarmonyReference.load(reference.json)`; M2 ranks against `logits_sorted` |
| Trend model | M3 and M4 | `_get_trend()` reuses `_get_recommender().trend_model` |
| SegFormer, CLIP, YOLO models | M1 (and M4's CLI path through M1) | Cached in Module 1 module globals |
| Settings | All of `demo.py` and M4 | One `M4Settings` object loaded from `m4_settings.json` |
| Area frame (98,304 px = 256 × 384) | M2, M4, feature-store builder | `outfit_state.REFERENCE_FRAME_PX`; user-photo areas are rescaled to it before scoring |

### 7.2.5 Can a module operate independently?

| Module | Independent in the GUI? | Independent from the command line? |
|---|---|---|
| M1 | Yes (choose "Module 1") | Yes (`run_pipeline.py`) |
| M2 | Needs M1's output and the M4 cache (shared reference) | Yes (`demo_score.py` uses its own 500-outfit reference) |
| M3 | Needs M1's output only; falls back to its own trend pickle if the M4 cache is missing | Yes (`render_trend_demo.py`) |
| M4 | Needs M1's output, the GNN checkpoint, the trend model and the cache | Yes (`run_recommend.py`) |

---

## 7.3 Data Flow Design

### 7.3.1 Request-time flow (one photo, "All" selected)

| Step | What happens | Function / file | Data form |
|---|---|---|---|
| 1 | The user uploads or pastes a photo. Gradio saves it to a temporary file and passes its **file path** | `gr.Image(type="filepath")` → `analyze(image_path, …)` | `str` path |
| 2 | Required assets are checked; Module 4 is loaded (first run only) | `analyze()`, `_get_recommender()` | — |
| 3 | Person/animal check: YOLO11-seg finds people and animals. No person but an animal stops the request. A pet next to a person is masked out | `subject_check.check_subjects()`, `decide()` | Boolean masks |
| 4 | SegFormer-B2-Clothes parses the photo into garment regions; CLIP gives each region a specific type and attributes | `fashion_segmenter.process_image_segformer()`, `attribute_analyzer.analyze_garment()` | In-memory PIL images and dicts |
| 5 | Crops, metadata and overlay are **written to disk** in `Fashion144k_v1/demo_out/demo_live/` | `process_image_segformer()` | PNG, JSON, JPG |
| 6 | A 512-D CLIP embedding is computed for every crop and **saved next to it** | `clip_extract_embeddings.extract_folder_embeddings(force_recompute=True)` | `.npy` (float32, 512) |
| 7 | Perception figure rendered | `render_perception_demo.render()` → `demos/perception_demo.png` | PNG |
| 8 | **M2**: the folder is read back, nodes are built, areas are rescaled to the 98,304-px frame, the GNN scores the outfit and a seeded one-garment corruption; both are ranked against the shared reference | `run_module2()` → `PhotoScorer.score_photo()` | Node dicts → PyTorch graph → float logit → percentile |
| 9 | **M3**: each garment's embedding is matched to the nearest cluster centroid of its broad category; below a cosine of 0.20 the garment counts as "no confident match" | `render_trend()` → `score_outfit_dir()` → `PerCategoryTrendModel.score_outfit()` | Dict of per-garment scores |
| 10 | **M4 state**: the folder is loaded into an `OutfitState`. For each piece: GNN input projection (128-D), normalised embedding, hue, formality, rescaled area, trend score, cluster and colour histogram | `outfit_state.load_outfit_dir()` | Frozen dataclasses with tensors |
| 11 | **Candidate retrieval** for each piece: allowed replacement types from the swap group → co-occurrence filter → representative pool rows of those types in the chosen pool mode → remove near-duplicates of the worn item → optional trend floor | `Recommender._candidates()`, `CandidatePool.candidate_rows()`, `dup_mask()`, `CooccurrenceFilter.allowed()` | GPU tensor of pool row indices |
| 12 | **What-if outfits**: the fixed pieces are repeated across a batch and only slot *i* is replaced with each candidate's precomputed projection, gathered in one indexing operation | `SwapScorer.swap_logits()` | Tensors `[B, n, 128]` and `[B, n]` |
| 13 | **Harmony of every candidate**: dense GNN forward pass in VRAM-sized chunks (halved and retried on out-of-memory) | `DenseGNN.logits_chunked()` | float32 logits |
| 14 | **Trend of every candidate**: looked up from the pool's precomputed `trend` column; ΔT = (candidate − worn) / σ_trend, set to 0 when either is unknown. ΔH = (logit − r₀) / σ_logit | `Recommender.score()` | `PieceSwaps` per piece → `SwapTable` |
| 15 | **Decision** (NumPy only, no GNN): combined gain, selection rule per piece, threshold τ, verdict swap / refresh / keep, top 3 suggestions from different style clusters | `Recommender.decide()`, `selection.py` | `Recommendation` dataclass |
| 16 | **Presentation**: verdict text, gallery of suggestion crops with captions, "also worth improving" line, per-piece table, swap dropdown, outfit board PNG | `demo_panel.*`, `outfit_board.outfit_board()` → `demos/recommendation_board.png` | Markdown strings, lists, PNG path |
| 17 | `analyze()` returns a 12-item tuple in the order of `OUTPUT_NAMES`. Gradio fills the widgets; the M4 state is kept in `gr.State` | `analyze()` | Tuple of paths, strings, lists, objects |

### 7.3.2 Exchange formats

| Boundary | Format |
|---|---|
| Browser → backend | Temporary image file path (Gradio) |
| M1 → M2 / M3 / M4 | Files on disk: `metadata.json` (list of dicts), `.npy` per garment, PNG crops, `<id>_vis.jpg` |
| Inside M4 | PyTorch tensors on the GPU (pool columns, projections), NumPy arrays (gains, rows), frozen dataclasses (`Piece`, `OutfitState`, `PieceSwaps`, `SwapTable`) |
| M4 → GUI | `Recommendation` and `Suggestion` dataclasses (they can also be written to JSON with `to_json()`) |
| Backend → browser | PNG file paths for figures, Markdown strings, a list of rows for the table, a list of (image path, caption) pairs for the gallery |

Not all data passes through JSON. JSON is used for Module 1's metadata, the settings, the store
indexes and some cache files. Embeddings and features are NumPy binary arrays. Within a module,
data passes as Python objects and tensors in memory.

### 7.3.3 Temporary, persistent and cached data

| Data | Lifetime | Location |
|---|---|---|
| Uploaded photo | Temporary (managed by Gradio) | Gradio temp directory |
| M1 output for the current photo | **Overwritten by each request** (fixed id `demo_live`) | `Fashion144k_v1/demo_out/demo_live/` |
| Result figures | Overwritten by each request | `demos/*.png` |
| Loaded models, pool tensors, reference | In memory for the life of the process | RAM / GPU memory |
| Scored candidate table (`SwapTable`) and `Recommendation` | One per browser session | `gr.State` |
| Feature stores, checkpoints, trend model, M4 cache | Persistent, built offline | See Section 7.4 |

**Request-level caching.** The most important cache inside a request is the `SwapTable`. It holds
the standardised ΔH and ΔT for every candidate. `Recommender.score()` is the only step that runs the
GNN. Moving the trend-weight slider or changing the selection rule calls `rerank()`, which calls only
`Recommender.decide()` on the stored table. This is pure NumPy, so the result updates immediately
without re-scoring. The pool mode, same-type and trend-floor options change which candidates are
scored, so they take effect on the next **Run**.

**Embeddings are not cached between requests.** Module 1 recomputes them for every photo
(`force_recompute=True`). Module 3 scores only the garments listed in the new `metadata.json`, so
stale files left in the reused folder are ignored.

---

## 7.4 Database and Data Storage Design

### 7.4.1 Storage approach

**The application uses no database.** No relational, document or key-value database is used at
inference time. All data lives in **local files**:

- datasets as image folders with CSV, MATLAB `.mat` and JSON metadata;
- garment features as **NumPy arrays** (`.npy`, opened memory-mapped) with a **JSON index**;
- model weights as a **PyTorch state dict** (`.pt`) and a **pickled Python object** (`.pkl`);
- the Module 4 cache as `.npy`, `.npz` and JSON files, with a fingerprint file that records when
  the cache is out of date.

The application keeps **no user, session or history records**. The only state for one browser
session is the in-memory `gr.State`, which is lost when the page or the process closes. The only
SQLite database in the repository belongs to the offline Pinterest scraper
(`PinterestData/pinterest_large_scale.py`, class `ProgressDB`). It records which search queries
have finished and which image hashes have been downloaded, so the scrape can resume after an
interruption. The application never reads it.

All of these assets are excluded from version control (`.gitignore` blocks datasets, feature
stores, caches, `.npy/.npz/.pt/.mat/.pkl` and every image format), so they exist only on the
machine that built them.

### 7.4.2 Storage inventory

| Data or Asset | Storage Format | Location / Directory | Used By | Purpose |
|---|---|---|---|---|
| Fashion144k photos | JPG + `photos.txt` index list | `Fashion144k_v1/photos/` | M1 (offline, `01_segment_batch.py`) | Source outfits for GNN training and the catalogue |
| Fashion144k split and votes | MATLAB `.mat` | `Fashion144k_v1/split.mat`, `Fashion144k_v1/feat/relvotes.mat` | M2 training, `PhotoScorer`, M4 cache | Train / val / test split; fashionability votes |
| Pinterest scrape | JPG + `metadata.csv` (image path, pin id, `date_ym`, board, query, …) | `PinterestData/pinterest_dataset/` | M3 (offline) | Dated images for trend analysis |
| Pinterest garment manifest | CSV + `.npy` (N × 512) | `PinterestData/store/garment_manifest.csv`, `garment_embeddings.npy` | M3 `fit_per_category.py` | One row per garment with date and category, for fitting the trend model |
| Pinterest split and queries | JSON `{key: split}`; CSV `key,query` | `PinterestData/store/split.json`, `outfit_queries.csv` | M4 cache | Split for evaluation; query-based demo exclusions |
| H&M product data | CSV (`articles.csv`, …) + product JPGs | `H&M Dataset/` | `Module4/src/hm_catalogue.py` (offline) | Studio product photos for the demo catalogue |
| H&M split / alignment record | JSON | `H&M Dataset/store/split.json`, `feature_store/alignment.json` | M4 cache | Split; records that CORAL alignment was applied |
| Segmented garment folders (one per outfit) | PNG crops + `.npy` embeddings + `metadata.json` + `outfit_features.json` | `Fashion144k_v1/outputs/<idx>/`, `PinterestData/store/outputs/<key>/`, `H&M Dataset/store/outputs/<key>/` | Feature-store builder; demo gallery (crop images) | M1 output for every catalogue item |
| **Feature stores** (×3) | `features.npy` float32 [N × 570]; `node_scalars.npy` float32 [N × 3]; `index.json` | `<dataset>/…/feature_store/` | M2 training; M4 cache and pool | Precomputed GNN node inputs for every garment |
| H&M raw features | `features_raw.npy` | `H&M Dataset/store/feature_store/` | `hm_catalogue.py --align` | Unaligned copy; every alignment run starts from it |
| SegFormer-B2-Clothes | Hugging Face weights | Downloaded to the Hugging Face cache (`mattmdjaga/segformer_b2_clothes`) | M1 | Garment parsing |
| CLIP ViT-B/32 | OpenAI CLIP weights | Downloaded by the `clip` package to its cache | M1, M3, M4 (crop gate) | Embeddings and zero-shot labels |
| YOLO11m-seg | Ultralytics `.pt` | `Module1/weights/yolo11m-seg.pt` (downloaded on first use) | M1 `subject_check.py` | Person / animal check |
| **Harmony GNN checkpoint** | PyTorch state dict `.pt` | `Fashion144k_v1/ck_attr_subset_s42/clip_outfit_gnn_best.pt` | M2, M4 | Trained compatibility model |
| **Trend model** | Pickled `PerCategoryTrendModel` `.pkl` | `Module3/per_category_trend_model_store.pkl` | M3, M4 | Per-category clusters, centroids and trend scores |
| Settings | JSON | `Module4/config/m4_settings.json` | `demo.py`, M4 | All paths and calibrated settings (`w_t`, rule, τ, ε, duplicate rule, crop gate, VRAM cap) |
| M4 pool metadata | `.npz` (12 per-row arrays) | `Module4/cache/pool/meta.npz` | `CandidatePool` | Source, split, type, broad category, outfit, crop present, trend, cluster, repost group, representatives |
| M4 precomputed projections | `.npy` float32 [690,604 × 128] | `Module4/cache/pool/proj.npy` | `SwapScorer` | GNN input projection for every catalogue garment |
| M4 file names / outfit keys | JSON | `Module4/cache/pool/files.json`, `outfit_keys.json` | `CandidatePool` | Map a pool row to its crop path and outfit |
| Co-occurrence matrix | `.npy` int64 [16 × 16] | `Module4/cache/pool/cooc.npy` | `CooccurrenceFilter` | Blocks garment-type combinations that do not occur in real outfits |
| Harmony reference | JSON (sorted logits, σ_logit, σ_trend) | `Module4/cache/pool/reference.json` | M2, M4 | Percentile scale and standardisation units |
| Cache fingerprint | JSON (SHA-256 prefixes, row counts) | `Module4/cache/pool/fingerprint.json` | `build_cache.py`, `CandidatePool.load` | Detects a stale cache |
| Crop quality metrics | `.npz` | `Module4/cache/crop_metrics.npz` | `crop_quality.crop_ok` | Demo-pool crop quality gate |
| Live request output | PNG / `.npy` / JSON / JPG | `Fashion144k_v1/demo_out/demo_live/` | M1 → M2, M3, M4 | Current photo's garments (overwritten) |
| Demo figures | PNG | `demos/` | Gradio outputs | Rendered results (overwritten) |

### 7.4.3 Feature-store design

All three catalogues use one feature-store layout, written by `Module2/scripts/build_feature_store.py`:

- **`features.npy`**: one float32 row of 570 values per garment: 512 CLIP, 32 colour histogram
  (16 hue + 8 saturation + 8 value bins), 26 encoded attributes. It is written as a memory-mapped
  NumPy array, so readers can load only the columns they need.
- **`node_scalars.npy`**: `[hue (NaN if unknown), formality, area]` per garment, used for the GNN's
  edge features. Areas are expressed in the 98,304-px frame, so garments from photos of different
  sizes compare fairly.
- **`index.json`**: the array dimensions and `outfits: {outfit_key: [{row, file, category,
  broad_category}, …]}`. This maps each row back to its outfit, its crop file and its type.

**Identifiers.** A garment is identified by *(store, outfit key, file name)*, e.g.
`("hm", "108775015", "t_shirt_0.png")`. Inside Module 4 it has a single **global pool row**: the
store's offset plus its row in that store. Fashion144k outfit keys are the dataset's integer
indices. Pinterest keys are integers given to the unique image files in path order. H&M keys are article ids without
the leading zero, and each H&M "outfit" is a single product.

| Store | Garment rows | Outfits | Role |
|---|---|---|---|
| Fashion144k | 480,079 | 144,169 | GNN training; evaluation pool; harmony reference |
| Pinterest | 154,663 | 55,781 | Trend source; evaluation pool; harmony reference |
| H&M | 55,862 | 55,862 (one product each) | **Demo suggestions** (`demo_sources: ["hm"]`) |

**Combining the stores.** The stores are not merged into one file. `CandidatePool` concatenates
them in memory, in the order listed in `m4_settings.json`, when it loads. Each row keeps a `source`
code, so every suggestion can name the catalogue it came from. The H&M product photos look
different from street photos (studio lighting, plain background), and CLIP picks this up in their
embeddings. `hm_catalogue.py --align` therefore maps each type's H&M embeddings onto the mean and
covariance of the same type in the street-photo stores (CORAL). It keeps the raw features in
`features_raw.npy` and records the method in `alignment.json`.

### 7.4.4 Retrieval, indexing and filtering

Module 4 looks up candidates with **boolean masks over in-memory tensors**, not a search index.
`CandidatePool.candidate_rows()` selects rows whose type is in the allowed set *and* that are a
representative in the chosen pool:

- **Representatives** (`rep_demo`, `rep_eval`): for each group of reposted outfits and each garment
  type, one row is kept (the largest crop). Reposted outfits, where the same photo appears more
  than once, are therefore counted once.
- **Eval pool**: test-split rows only (garments the GNN never trained on).
- **Demo pool**: worked out again each time the pool loads (`gate_demo_pool`): only stores listed
  in `demo_sources`, only rows whose crop exists on disk, only crops that pass the quality gate
  (`crop_gate` in settings), and no outfits found by an excluded search query. Because this filter
  runs at load time, changing these settings does not require rebuilding the cache.

Further per-piece filters are the swap group (`swap_groups.replacement_types`), the co-occurrence
matrix (`cooc_min = 20`), the near-duplicate rule (CLIP cosine ≥ 0.99 **and** colour-histogram
L1 ≤ 0.25 **and** area ratio ≤ 1.25) and the optional trend floor.

### 7.4.5 Model storage and loading

| Model | Source | Saved as | Loaded by |
|---|---|---|---|
| SegFormer-B2-Clothes | Downloaded pretrained (Hugging Face) | HF cache | `AutoModelForSemanticSegmentation.from_pretrained()` |
| CLIP ViT-B/32 | Downloaded pretrained (OpenAI) | `clip` cache | `clip.load("ViT-B/32")` |
| YOLO11m-seg | Downloaded pretrained (Ultralytics) | `Module1/weights/` | `ultralytics.YOLO(WEIGHTS)` |
| Harmony GNN | **Trained in this project** (`Module2/src/03_train_gnn.py`) | `state_dict` `.pt` per epoch and `_best.pt` | `CLIPOutfitGNN(...)` + `load_state_dict(torch.load(..., weights_only=True))` |
| Trend model | **Fitted in this project** (`Module3/scripts/fit_per_category.py`) | Whole object pickled | `pickle.load` via `PerCategoryTrendModel.load()` |

The GNN's architecture is not stored in the checkpoint. It is rebuilt in code (`attr_dim` = 11
selected attributes, `edge_dim` = 5) and the weights are then loaded into it. The preprocessing
needed to read a checkpoint correctly (which attribute columns, which area frame) is defined in
code (`m2_bridge.ATTR_MASK`, `FEATURE_COLS`, `REFERENCE_FRAME_PX`). The Module 4 settings are the
only separately stored configuration.

### 7.4.6 Cache validity

`build_cache.py` writes a **fingerprint** of everything the cache depends on: SHA-256 prefixes of
the checkpoint, the trend model and each split file, each store's row count and output-folder
count, the H&M features hash (because alignment rewrites them) and the duplicate-rule settings.
`CandidatePool.load()` refuses a missing or mismatched cache with a message telling the user to run
`build_cache.py --force`. While rebuilding, the old fingerprint is deleted first, so an interrupted
build can never be mistaken for an up-to-date one.

---

## 7.5 Interface Design

### 7.5.1 Layout

`build_ui()` in `demo.py` builds one page (`gr.Blocks`, title "Outfit Compatibility Demo") with a
short explanation at the top and two columns below it:

- **Left column (inputs, 1/3 width):** photo input, module selector, device selector, trend-weight
  slider, a collapsed "Module 4 advanced options" section, the **Run** button and an example photo.
- **Right column (outputs, 2/3 width):** the Module 1, 2 and 3 figures, then the Module 4 section
  (verdict, suggestion gallery, "also worth improving", swap preview, per-piece table) and a log
  box.

There are no tabs. Every output widget is on the page at all times. Modules that were not run show
empty widgets.

### 7.5.2 Interface elements and backend functions

| Element | Gradio component | Options / behaviour | Backend |
|---|---|---|---|
| Outfit photo | `gr.Image(sources=["upload","clipboard"], type="filepath")` | Upload or paste. The code sets no file-type restriction; Module 1's batch mode accepts .jpg .jpeg .png .webp .bmp | Input to `analyze` |
| Which module to run | `gr.Radio` | Module 1 / 2 / 3 / 4 / All (default All) | `analyze` |
| Device | `gr.Radio` | auto / cuda / cpu; takes effect on the first run only, because models load once | `analyze` |
| Trend weight | `gr.Slider(0–1, step 0.05)` | Default `w_t` from settings (0.2); w_h = 1 − w_t | `analyze`; on release → `rerank` |
| Candidate pool | `gr.Radio` (advanced) | demo / eval | `analyze` → `Recommender.score` |
| Same type only | `gr.Checkbox` (advanced) | Strict like-for-like swaps | `analyze` |
| Trend floor | `gr.Checkbox` (advanced) | Never suggest a less trendy piece | `analyze` |
| Selection rule | `gr.Dropdown` (advanced) | max_gain / topk_mean_gain / slot_percentile | `analyze`; on change → `rerank` |
| Run | `gr.Button` | Starts processing | `run_btn.click(analyze)` |
| Example | `gr.Examples` | `samples/image2.jpg` (shown only if the file exists) | Fills the photo input |
| Module 1 figure | `gr.Image` | Segmentation overlay and attributes | `render_perception_demo.render` |
| Module 2 figure | `gr.Image` | Harmony score figure | `PhotoScorer.score_photo` |
| Module 3 figure | `gr.Image` | Per-garment trend figure | `render_trend_demo.render` |
| Verdict | `gr.Markdown` | Headline, harmony %, mean trend, weight, rule, pool, notes | `demo_panel.verdict_markdown` |
| Suggestions | `gr.Gallery(columns=3)` | Crop image and caption (type change, harmony and trend before → after, source, "what changed") | `demo_panel.gallery_items` |
| Also worth improving | `gr.Markdown` | Other pieces that also clear τ | `demo_panel.also_markdown` |
| Try this swap | `gr.Dropdown` | One entry per suggestion | On change → `try_swap` |
| Outfit with the swap | `gr.Image` + `gr.Markdown` | Outfit board with the new piece highlighted, plus a one-line summary | `demo_panel.swap_board` → `outfit_board` |
| Per piece | `gr.Dataframe` (read-only) | piece, type, candidates, selection score, best ΔH, best ΔT, clears τ, notes | `demo_panel.per_piece_rows` |
| Log | `gr.Textbox(lines=10)` | Step-by-step progress, scores, timings, errors | Built in `analyze` |
| Session state | `gr.State` | `{"table": SwapTable, "rec": Recommendation}` | Written by `analyze` / `rerank`; read by `rerank`, `try_swap` |

### 7.5.3 Event wiring

| Event | Handler | Inputs | Outputs | Runs GNN? |
|---|---|---|---|---|
| `run_btn.click` | `analyze` | photo, module, device, pool, same type, trend floor, rule, w_t | 3 figures, log, 8 Module 4 outputs | Yes |
| `w_t.release` | `rerank` | state, w_t, rule | 8 Module 4 outputs | No |
| `rule.change` | `rerank` | state, w_t, rule | 8 Module 4 outputs | No |
| `choice.change` | `try_swap` | state, selected label | board image, board text | No |

The Module 4 widgets are produced by `Module4/src/demo_panel.py`, which does not import Gradio. It
converts a `Recommendation` into plain strings, lists and image paths, and `demo.py` only places
those values into widgets. The presentation logic can therefore be unit-tested without starting the
interface (`test_demo_panel.py`).

### 7.5.4 Error and invalid-input handling

| Situation | Behaviour |
|---|---|
| No photo | "Please upload or paste an outfit photo first." |
| Missing checkpoint, trend model or M4 cache | A message naming the missing file and the command that builds it |
| Photo of an animal with no person | Module 1 returns `animal_no_person`; processing stops with "No person found, only an animal…" |
| No garments detected | "No garments found in this photo. Try a clearer, full-body photo of a person." |
| Stale M4 cache | `CandidatePool.load` raises an error asking for `build_cache.py --force`; shown in the log |
| Any other exception | Caught in `analyze`; the error and the full traceback go to the Log box; the outputs are cleared |
| Slider / rule moved before any run | "Run Module 4 first." |
| Swap option no longer listed | "That option is no longer listed: press Run again." |
| Outfit already strong | Verdict "keep" (or "refresh"); the board shows "Nothing to try: the outfit is already strong." |

---

## 7.6 Design Decisions and Trade-offs

Each decision below is marked **[Documented]** if the code or project documents state the reason,
or **[Inferred]** if the reason is read from the implementation.

| # | Decision | Benefit | Limitation / trade-off |
|---|---|---|---|
| 1 | **Local, single-process execution** on one Windows machine, with a 6 GB VRAM cap (`apply_vram_cap`) **[Documented: cap and shared GPU in `m2_bridge.py`; local run in README]** | No hosting cost, no network delay, data stays on the machine, simple deployment | Needs a CUDA GPU for interactive speed (a CPU fallback works but is slow); assets total several GB and exist on only one machine; not set up for many users |
| 2 | **Gradio as the presentation layer**, with presentation code kept separately in `demo_panel.py` **[Inferred]** | Quick to build with Python only; built-in image, gallery and table widgets; the presentation logic is testable without the UI | Limited control over layout; the fixed request id `demo_live` and fixed figure paths mean simultaneous users would overwrite each other's files, so the interface suits one user at a time **[Inferred]** |
| 3 | **Four separate modules, one folder each, connected through Module 1's output folder** **[Documented: README "How the modules connect"]** | Modules can be developed, tested and run separately from the command line; any module can be swapped out if it keeps the folder format | The folder must be written and read back on every request (small disk overhead); there are two GNN scoring paths (Module 2's graph builder and Module 4's dense version), which need a test to keep them equal |
| 4 | **One shared garment representation** (CLIP + colour + attributes) and **one feature-store format** for all catalogues **[Documented: `build_feature_store.py`, `hm_catalogue.py`]** | A user's garment and catalogue garments are directly comparable; a new catalogue only needs to go through the same Module 1 → store steps | Everything is tied to CLIP ViT-B/32 and the 570-D layout; changing the embedding means rebuilding every store, the GNN and the trend model |
| 5 | **Precompute catalogue features** (stores) and **GNN input projections** (`proj.npy`) offline **[Documented: `candidate_pool.py`, `dense_gnn.py`]** | No catalogue image is processed at request time; building all what-if outfits becomes one indexing operation | 338 MB projection file plus about 1.8 GB of feature-store files; the cache must be rebuilt whenever the checkpoint, trend model or stores change |
| 6 | **Load trained models and cached assets; never train during inference** **[Documented]** | Predictable response time after the first run | The first request is slow (about 30–60 s); the device choice is fixed after the first load |
| 7 | **Files instead of a database** **[Inferred]** | Fits read-only, array-shaped data; memory-mapped NumPy arrays load fast; no database server to install | No concurrent writes, no queries beyond in-memory masks, no record of users or sessions; consistency depends on the fingerprint check |
| 8 | **Fingerprinted cache** with automatic stale detection **[Documented: `pool_fingerprint`]** | Prevents silently scoring with a mismatched model and cache | Any change to the checkpoint, trend model or stores requires a full rebuild |
| 9 | **Module 4 reuses Module 2's trained GNN** (through `m2_bridge.py`) instead of a separate model, and **shares one instance** with Module 2 in the app **[Documented: `demo.py` docstring]** | One harmony judgement for the whole system; Modules 2 and 4 report the same harmony % ; one GNN in memory | Module 2 in the GUI depends on the Module 4 cache; Module 4 inherits any bias the GNN has |
| 10 | **Batched, vectorised candidate scoring**: every allowed candidate is scored, not a sample **[Documented: module4-design.md §5]** | No sampling noise; the design target was ≤ 5 s per photo, and the measured time is 0.16 s using under 2 GB (module4-design.md) | Needs GPU memory; handled by VRAM-based chunking and halving the chunk on out-of-memory |
| 11 | **Score once, re-decide many times** (`score()` vs `decide()`) **[Documented: `recommender.py`]** | The slider and rule changes respond instantly; evaluation sweeps reuse one scoring pass | The table is held in session memory; changing the pool or filters still needs a full Run |
| 12 | **Combining Fashion144k, Pinterest and H&M**: Fashion144k and Pinterest for evaluation, the harmony reference and co-occurrence; **H&M product photos for the suggestions shown in the demo**, after CORAL alignment **[Documented: `hm_catalogue.py`, `m4_settings.json`]** | Clean product images make readable suggestions; Pinterest provides the current trend signal; honest evaluation on the test split | Studio photos differ from street photos, which must be corrected (CORAL); H&M items are single products, so they add nothing to the co-occurrence matrix or the harmony reference |
| 13 | **Settings in one JSON file with calibration status** (`m4_settings.json`) **[Documented]** | Tuned values (w_t, rule, τ, ε, t_dup) can be changed without code changes; any uncalibrated value is shown as "provisional" in every result | Settings and cache must be kept consistent (the fingerprint covers the duplicate rule only) |
| 14 | **Large files kept out of Git** (`.gitignore`) **[Documented]** | The repository stays small; no images are committed | The system cannot be reproduced from the repository alone; datasets, checkpoints and caches must be copied or rebuilt |

---

## Suggested figures for Chapter 7

1. **Figure 7.1 – Offline and online phases.** Two swim-lanes. Offline: datasets → M1 batch
   segmentation → garment folders → feature stores → GNN training / trend fitting / H&M alignment
   → `build_cache.py` → `Module4/cache/pool`. Online: photo → `demo.py` → M1 → M2 / M3 / M4 → UI.
   Dashed arrows show which offline asset each online step reads.
2. **Figure 7.2 – Module interaction diagram.** The four modules as boxes, with arrows labelled by
   mechanism: "files: metadata.json + .npy" (M1→M2/M3/M4), "import via m2_bridge: GNN" (M2→M4),
   "pickle: PerCategoryTrendModel" (M3→M4), "Recommendation object" (M4→UI). Mark the shared GNN
   instance and the shared harmony reference.
3. **Figure 7.3 – Request sequence diagram.** User → Gradio → `analyze` → `run_single_image` →
   `score_photo` → `render_trend` → `load_outfit_dir` → `Recommender.score` → `decide` →
   `demo_panel` → Gradio. Show the early exits (animal only, no garments) and a second sequence for
   the slider → `rerank` → `decide`.
4. **Figure 7.4 – Storage layout.** A directory tree of the persistent assets (datasets, the three
   feature stores with their three files, checkpoint, trend pickle, `Module4/cache/pool/` files) and
   the per-request folder `demo_out/demo_live/`.
5. **Figure 7.5 – Annotated UI screenshot** with callouts mapping each widget to its callback
   (Table 7.5.2).
