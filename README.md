# AI-Powered Personal Stylist




Take a photo of an outfit and the system tells you **what you're wearing**, **how well it goes
together**, **how on-trend each piece is**, and **which one piece to change, and to what**.

```
 photo
   │
   ▼
 Module 1  Perception      "What's in the photo?"            → each garment cut out and described
   │
   ├──► Module 2  Harmony  "Do these pieces go together?"    → harmony score (0–100 percentile)
   ├──► Module 3  Trend    "Is each piece popular now?"       → trend score per garment (0–100)
   ▼
 Module 4  Recommendation  "What one swap improves it most?"  → piece to swap + 3 suggestions
```

## Modules

| Module | What it does | Status | Read more |
|---|---|---|---|
| **1. Perception** | Segments the photo into garments, names them, and describes each one (colour, silhouette, formality, …) with a CLIP embedding | ✅ done | [`Module1/README.md`](Module1/README.md) |
| **2. Harmony** | A graph neural network scores how well the pieces go together. Trained on ~144k real outfits | ✅ done (test AUC ≈ 0.80) | [`Module2/README.md`](Module2/README.md) |
| **3. Trend** | Scores each garment by how present its style is in recent Pinterest activity | ✅ done | [`Module3/README.md`](Module3/README.md) |
| **4. Recommendation** | Tries replacements for each piece and recommends the swap that helps most, or says "keep it" | 📝 designed, not built | [`Module4/README.md`](Module4/README.md) |

## Try it

```bash
pip install -r Module1/requirements.txt
python demo.py
```
A web app opens in your browser. Upload an outfit photo, choose a module (or **All**), and click **Run**.
More in [`demos/README.md`](demos/README.md).

The demo needs the trained models and data locally (see **Data** below). The first run loads models
(~30–60 s); later runs are fast.

## How the modules connect

| Producer | Produces | Consumed by | Why |
|---|---|---|---|
| Module 1 | Per garment: transparent cut-out, 512-D **CLIP embedding**, attributes, category | Modules 2, 3, 4 | The shared description of an outfit |
| Module 1 (run offline over datasets) | **Feature store**: hundreds of thousands of garments, already described | Modules 2, 4 | Training data for M2; the catalogue of replacements for M4 |
| Module 2 | **Harmony score** for a whole outfit | Module 4 | Judges every "what-if" outfit |
| Module 3 | **Trend score** + style cluster per garment | Module 4 | Judges how current a replacement is; gives variety among suggestions |

## Repository layout

```
demo.py              web demo (Gradio) for all modules
Module1/ … Module4/  one folder per module, each with its own README
PinterestData/       Pinterest scraper scripts + dataset notes (the data itself is not in git)
docs/                project docs: GNN training, Module 4 design, next steps
demos/               where the demo writes its figures (images are not in git)
samples/             example photos (local only, not in git)
```

## Docs

- [How the harmony GNN was trained](docs/gnn-training.md): approach, results, problems fixed, limits
- [Module 4 design (plain language)](docs/module4-design.md) and the [detailed spec](docs/module4-design-detailed.md)
- [Next steps & backlog](docs/next-steps.md)
- [Pinterest dataset notes](PinterestData/README_DATASET.md)

## Data (not in git)

Datasets, trained models and generated files are large, so they're **gitignored** and live only on
the training machine:

| Folder | Contents |
|---|---|
| `Fashion144k_v1/` | Fashion144k photos, splits, segmentation outputs, feature store, GNN checkpoints |
| `PinterestData/pinterest_dataset/` | Scraped Pinterest images + `metadata.csv` |
| `PinterestData/garment_data/` | Pinterest garments: manifest + embeddings |
| `Module3/per_category_trend_model.pkl` | The fitted trend model |

**Rule: no images are ever committed.** `.gitignore` blocks every image format. Figures are regenerated
by the demo, and sample photos stay in `samples/`.

## Glossary

| Term | Meaning |
|---|---|
| **Segmentation (SegFormer)** | Labelling every pixel of a photo ("shirt", "pants", "background") to cut out each garment |
| **CLIP** | A model that turns an image into 512 numbers (an **embedding**) capturing how it looks, and can match images to text |
| **Embedding** | A list of numbers representing an image. Similar-looking garments get similar numbers |
| **Zero-shot** | Classifying with text prompts ("a jacket") without extra training |
| **HSV** | Hue / Saturation / Value, a colour description that's easy to reason about |
| **Graph / node / edge** | An outfit as dots (garments) joined by lines (how two garments relate) |
| **GNN** | Graph neural network: garments "pass messages" to each other, so each one is judged in context |
| **BPR loss** | A training goal: "the real outfit should score higher than a spoiled copy" |
| **Negative sampling** | Making a deliberately worse outfit (one garment swapped) to compare against |
| **AUC** | How often the model ranks real above spoiled: 0.5 = guessing, 1.0 = perfect |
| **Ablation** | Removing a component to measure what it contributes |
| **Percentile** | Where a score falls among real outfits (80 = better than 80% of them) |
| **K-Means / style cluster** | Grouping similar-looking garments. Each group is a "style" whose popularity is tracked over time |
| **Feature store** | All garments' numbers precomputed into one file, so nothing is recomputed |
