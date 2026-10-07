# Pinterest Fashion Dataset

Source data for the trend model (Module 3) of the AI Personal Stylist project.
In this repository the data root is `PinterestData/` (the data itself is gitignored); paths below use
`<data_root>` as a placeholder.

---

## Overview

This dataset supports the **Trend Intelligence Module** of the AI Personal Stylist project. It contains fashion images scraped from Pinterest along with their creation timestamps, which are used to construct temporal trend curves for each discovered style cluster.

The dataset feeds directly into:
- **K-Means clustering** over CLIP embeddings to discover style archetypes
- **Monthly frequency analysis** per cluster to model rising/declining trends
- **Candidate retrieval** for outfit replacement recommendations

---

## Folder Structure

```
<data_root>\
│
├── pinterest_data/                   ← RAW SCRAPE (intermediate, can delete raw/ after build)
│   ├── progress.db                   ← SQLite tracker (which queries completed)
│   ├── gdl_config.json               ← gallery-dl configuration used during scraping
│   └── raw/
│       └── pinterest/
│           └── Search/
│               ├── street+style+fashion/
│               │   ├── pinterest_2533343533223371.jpg
│               │   ├── pinterest_2533343533223371.jpg.json   ← metadata sidecar
│               │   └── ...
│               ├── minimalist+outfit+aesthetic/
│               │   └── ...
│               └── ... (132 query folders total)
│
└── pinterest_dataset/                ← FINAL DATASET (used by the project)
    ├── metadata.csv                  ← master index of all images
    └── images/
        ├── 2533343533223371.jpg      ← 224×224 RGB JPEG, CLIP-ready
        ├── 844493675839005.jpg
        └── ... (~61,000+ images)
```

---

## How the Dataset Was Built

### Step 1 — Scraping (gallery-dl)

Images are downloaded using **gallery-dl 1.31.10**, an open-source tool that interfaces with Pinterest's internal search API. For each curated fashion search query, gallery-dl:

1. Authenticates using Pinterest account credentials (in.pinterest.com region)
2. Paginates through search results up to 1,000 pins per query
3. Downloads the highest-resolution available image for each pin
4. Writes a `.json` sidecar file alongside each image containing full Pinterest pin metadata

Scraping is parallelised across 3 worker threads with per-query delays to avoid rate limiting. A SQLite database (`progress.db`) tracks completed queries, making the process resumable across sessions — safe to interrupt and rerun without re-downloading finished queries.

**Queries run:** 132

### Step 2 — Processing (build step)

Raw downloads are processed by `pinterest_large_scale.py build`:

1. **Deduplication** — MD5 hash of every image file; duplicate repins are discarded
2. **Size filtering** — images smaller than 100×100px are rejected
3. **Resizing** — all images resized to 224×224 JPEG using Lanczos resampling (matches CLIP ViT-B/32 input size)
4. **Metadata extraction** — date, query, board, description pulled from `.json` sidecars
5. **Date parsing** — Pinterest's RFC date format (`Sun, 25 Sep 2022 17:36:26 +0000`) normalised to `YYYY-MM`

**Final unique images:** 61,118  
**Images with valid date:** 100%  
**Date range:** 2011-10 → 2026-09

---

## Search Queries Used

132 queries were chosen to ensure broad stylistic, temporal, and cultural diversity:

| Category | Example Queries |
|---|---|
| Core outfit styles | outfit of the day women/men, street style fashion, casual chic, business casual |
| Aesthetics | minimalist, y2k, cottagecore, dark academia, old money, clean girl, boho, grunge, quiet luxury, mob wife, coquette, gorpcore, balletcore, office siren, dopamine dressing, scandi minimalist, indie sleaze |
| Garment combos | jeans and blazer, midi skirt, trench coat, wide leg pants, cargo pants, maxi dress, low rise pants, sheer layering, platform loafers, denim on denim |
| Occasion-based | wedding guest, festival fashion, airport outfit, concert outfit, athleisure, vacation resort |
| Color/pattern | monochrome, all black, neutral tones, pastel, floral print, animal print, color blocking |
| Trend-tagged | fashion trends 2026, viral tiktok fashion, pinterest fashion 2026, instagram baddie outfit |
| International | paris/new york/milan/tokyo/copenhagen/seoul fashion week street style |
| Indian ethnic wear | saree draping styles, lehenga choli, anarkali suit, salwar kameez, kurta set, kurti, palazzo suit, sharara suit |
| Indian fusion | indo western outfit, fusion wear saree, crop top with lehenga, dhoti pants, jacket lehenga |
| Indian occasion wear | mehendi, sangeet, haldi, diwali, navratri, indian wedding guest/groom, reception, engagement |
| Indian regional styles | banarasi saree, kanjivaram saree, south indian silk saree, punjabi suit, rajasthani traditional, bengali saree |
| Indian fashion scene | lakme fashion week, india couture week, bollywood airport look, indian influencer fashion, sustainable indian fashion brands |

Trend-tagged, occasion-based, and year-specific queries are included deliberately to pull content with known temporal context, ensuring a wide spread of `date_ym` values across the dataset. Indian fashion queries add cultural and stylistic diversity not represented in the core global-fashion query set.

---

## metadata.csv — Column Reference

The master CSV lives at `pinterest_dataset/metadata.csv`. Each row is one unique image.

| Column | Type | Example | Description |
|---|---|---|---|
| `image_path` | string | `images/2533343533223371.jpg` | Relative path from `pinterest_dataset/` root |
| `pin_id` | string | `2533343533223371` | Pinterest pin ID |
| `date_ym` | string | `2022-09` | Pin creation month — **primary field for trend curves** |
| `date_raw` | string | `Sun, 25 Sep 2022 17:36:26 +0000` | Raw Pinterest timestamp (for debugging) |
| `query` | string | `street style fashion` | Search query that sourced this pin |
| `board` | string | `vestidos que quiero hacer` | Pinterest board the pin was saved to |
| `description` | string | `Bella Hadid Style, Looks Street Style...` | Pin description or visual annotation tags |
| `orig_width` | int | `1280` | Original image width before resizing |
| `orig_height` | int | `1960` | Original image height before resizing |
| `hash` | string | `7611ab35701f3761c50a78369e04f6bd` | MD5 hash used for deduplication |

### Key fields for each module

- **Trend module** → `image_path` + `date_ym`
- **K-Means clustering** → `image_path` (embed all images with CLIP, cluster in embedding space)
- **Candidate retrieval** → `image_path` + `query` + `date_ym`
- **Deduplication/QA** → `hash` + `orig_width` + `orig_height`

---

## Sidecar JSON Structure

Each raw image in `pinterest_data/raw/` has a corresponding `.json` sidecar written by gallery-dl. Key fields used by the build script:

```json
{
    "id": "2533343533223371",
    "created_at": "Sun, 25 Sep 2022 17:36:26 +0000",
    "description": "",
    "pin_join": {
        "visual_annotation": ["Bella Hadid Style", "Looks Street Style", "Mode Inspo"]
    },
    "board": {
        "name": "vestidos que quiero hacer"
    },
    "search": "street+style+fashion",
    "width": 1280,
    "height": 1960,
    "domain": "vogue.com"
}
```

When `description` is empty, the build script falls back to `pin_join.visual_annotation` tags — these are Pinterest's computer-vision generated labels and are often more useful than user-written descriptions.

---

## Loading the Dataset

```python
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import clip

DATASET_DIR = Path("<data_root>/pinterest_dataset")

# Load master index
df = pd.read_csv(DATASET_DIR / "metadata.csv")

# Subset to dated images only (for trend curve construction)
df_dated = df[df["date_ym"].notna()].copy()
print(f"Total images:        {len(df)}")
print(f"Dated images:        {len(df_dated)}")

# Monthly distribution (visualise before running K-Means)
monthly = df_dated.groupby("date_ym").size().sort_index()
print(monthly.tail(12))

# CLIP embedding loop
model, preprocess = clip.load("ViT-B/32")
model.eval()

embeddings = []
for _, row in df_dated.iterrows():
    img_path = DATASET_DIR / row["image_path"]
    img = preprocess(Image.open(img_path)).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(img).squeeze().numpy()
    embeddings.append(emb)

# embeddings[i] corresponds to df_dated.iloc[i]
# Pass to K-Means, then use df_dated["date_ym"] for trend curves
```

---

## Detailed Analysis

For a full breakdown beyond the quick `stats` command (every query, every month across the full date range, plus a query × month heatmap), use `pinterest_dataset_analysis.py`:

```
python pinterest_dataset_analysis.py --dataset-dir <data_root>\pinterest_dataset
```

This produces:
- `monthly_distribution.png` — full-range monthly bar chart (2011-10 → 2026-09, including zero-count months)
- `query_breakdown.png` — image count per query, all 132 queries
- `query_monthly_heatmap.png` — query × month grid, useful for spotting which queries drive which time periods
- `analysis_summary.txt` — text version of the full breakdown

---

## Disk Space

| Folder | Contents | Approximate Size |
|---|---|---|
| `pinterest_data/raw/` | Original downloaded images + JSON sidecars | ~15–18 GB |
| `pinterest_dataset/images/` | Resized 224×224 JPEGs | ~7–8 GB |
| `pinterest_dataset/metadata.csv` | Master index | ~18 MB |

Once you have verified `pinterest_dataset` is complete and `metadata.csv` looks correct, you can safely delete `pinterest_data/raw/` to recover disk space. Keep `pinterest_data/progress.db` if you plan to scrape additional queries later.

```
rmdir /s /q <data_root>\pinterest_data\raw
```

---

## Adding More Images

The scraper is fully resumable. To add more images for existing or new queries:

```
python pinterest_large_scale.py scrape --max-per-query 500 --workers 3 --cookies cookies.txt --output-dir <data_root>\pinterest_data --queries "your query here" "another query"
```

Completed queries are skipped automatically based on exact query-string match in `progress.db`. After scraping, re-run build:

```
python pinterest_large_scale.py build --raw-dir <data_root>\pinterest_data\raw --dataset-dir <data_root>\pinterest_dataset
```

**Note:** using cookie-based auth (`--cookies cookies.txt`) rather than `--username`/`--password` avoids storing plaintext credentials on disk in `gdl_config.json` and is more stable against Pinterest's bot detection when running multiple parallel workers.

---

## Tools & Dependencies

| Tool | Version | Purpose |
|---|---|---|
| gallery-dl | 1.31.10 | Pinterest scraping and metadata extraction |
| Pillow | latest | Image resizing and format conversion |
| pandas | latest | CSV construction and dataset management |
| matplotlib | latest | Analysis charts |
| tqdm | latest | Progress bars |
| SQLite3 | built-in | Resumable progress tracking |
| Python | 3.12.6 | Runtime |
