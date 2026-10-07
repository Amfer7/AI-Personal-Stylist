# Module 1: Perception

**In one line:** turns an outfit photo into a clean cut-out of each garment, plus numbers and
descriptions a computer can reason about.

```
photo ──► find each garment ──► name it precisely ──► describe + fingerprint it ──► outputs/<id>/
          (SegFormer)            (CLIP zero-shot)      (attributes + CLIP embedding)
```

Every later module reads Module 1's output. Module 2 scores outfits from it, Module 3 scores trends
from it, and Module 4 builds "what-if" outfits from it.

---

## How it works

### 1. Find and cut out each garment
**SegFormer** (`mattmdjaga/segformer_b2_clothes`) labels every pixel of the photo: upper clothes, pants,
skirt, dress, shoes, bag, hat, belt, scarf, sunglasses, or skin/hair/background (discarded). This is
called **semantic segmentation**. Each clothing region is cut out as a **transparent PNG** (the alpha
channel marks the fabric pixels), so later steps see only the garment.
`bg_removal.py` handles the cut-out, with **GrabCut** as a fallback for small accessories.

> One garment per region: a jacket worn over a t-shirt comes out as **one** upper-body garment.

### 2. Name each garment precisely
SegFormer only says "upper clothes". **CLIP** compares the cut-out with text prompts ("a t-shirt",
"a sweater", "a jacket"…) and picks the best match. This is **zero-shot classification**: no extra
training needed. It produces 16 specific types (t_shirt, sweater, shirt_blouse, jacket, top, pants,
leggings, shorts, skirt, dress, shoe, bag, hat, belt, scarf, sunglasses), each with a **broad category**
(top / bottom / dress / accessory). Names are kept tidy by `category_mapping.py`.

### 3. Describe and fingerprint each garment
- **Embedding (the fingerprint):** CLIP ViT-B/32 turns the cut-out into **512 numbers**, normalised to
  length 1 (**L2-normalised**). Garments that look alike get similar numbers.
- **Attributes (the description)** from `attribute_analyzer.py`:
  - *measured from pixels:* dominant **colour** (RGB + **HSV**, via k-means over fabric pixels only)
    and **silhouette** (fitted / boxy / A-line / tapered, from the mask's shape);
  - *guessed by CLIP zero-shot (noisier):* pattern, fabric, fit, formality, and a 0–100 formality score.
- **Outfit-level "fashion-theory" features:** hue distance, volume balance, formality coherence.

## What it saves (the output contract)

```
outputs/<id>/
  <garment>_<n>.png      transparent cut-out
  <garment>_<n>.npy      512-D CLIP embedding
  metadata.json          per garment: category, broad_category, bbox, confidence, attributes
  outfit_features.json   outfit-level hue / volume / formality numbers
  <id>_vis.jpg           annotated preview
```
Other modules depend on this exact layout. Don't change it without updating them.

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py --image ../samples/image2.jpg --output_dir outputs --device cuda
python run_pipeline.py --input_dir path/to/photos --output_dir outputs   # a whole folder
python render_perception_demo.py --outfit_dir outputs/<id> --photo ../samples/image2.jpg --out perception_demo.png
python test_pipeline.py                                                   # checks
```
The easiest way to see it is the repo-root web demo: `python demo.py`, then choose **Module 1**.

## Files

| File | Role |
|---|---|
| `run_pipeline.py` | Entry point: one image or a folder → `outputs/<id>/` |
| `fashion_segmenter.py` | SegFormer segmentation + CLIP subtype classification |
| `bg_removal.py` | Transparent cut-outs (mask-based, GrabCut fallback) |
| `clip_extract_embeddings.py` | 512-D CLIP embeddings |
| `attribute_analyzer.py` | Colour, silhouette, pattern, fabric, fit, formality + outfit features |
| `category_mapping.py` | Category names and broad categories |
| `render_perception_demo.py` | Slide figure: photo → garments + attributes |
| `test_pipeline.py` | Checks (512-D unit-length embeddings, colour ignores background, …) |
| `yolo_detect_and_crop.py` | Older YOLO-based detector, kept for reference (not the default) |
| `changes.md` | Log of the attribute work added to this module |
| `PIPELINE_HANDOFF_GUIDE.md` | Historical handoff notes from the original pipeline |

**Note:** `Module3/` keeps identical copies of `fashion_segmenter.py`, `attribute_analyzer.py` and
`clip_extract_embeddings.py`. `Module3/test_module1_sync.py` fails if they drift, so change both together.

## Known limits
- One garment per body region, so layering isn't represented.
- CLIP's zero-shot guesses (pattern, fabric, fit) are noisy. Module 2 drops them (see
  [`docs/gnn-training.md`](../docs/gnn-training.md)).
