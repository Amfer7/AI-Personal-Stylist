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

### 0. Check who is in the photo (live photos only)
`src/subject_check.py` runs **YOLO11m-seg** (COCO instance segmentation, CPU, ~0.3 s) before anything else.
SegFormer assumes a person: on an undressed golden retriever it labelled the dog's face "shirt_blouse"
and the outfit scored harmony 100/100.
- **An animal and no person:** the pipeline stops with "No person found, only an animal" and writes no
  garments.
- **A person and an animal:** pixels on the animal (grown by 1% of the photo's longer side) but not on a
  person are treated as background, so a pet's sweater is not counted as the owner's clothes.
- **Otherwise:** unchanged. Flat-lays and product shots are not blocked.

Measured on 16 Unsplash photos (`samples/nonhuman/`, local only, sources in its `manifest.csv`): all 13
animal-only photos stop; on 600 real outfit photos (300 Fashion144k, 300 Pinterest) it never fires. A
whole-image CLIP "is this a person?" score was rejected: it put people in elaborate garments (saree,
lehenga, sherwani) below 0.3. Species names are not shown to the user, since COCO called a cat in a suit
a "bear". Only `run_single_image()` runs this check; feature-store builds call
`process_image_segformer()` directly and are unchanged. Weights download to `Module1/weights/` on first use.

### 1. Find and cut out each garment
**SegFormer** (`mattmdjaga/segformer_b2_clothes`) labels every pixel of the photo: upper clothes, pants,
skirt, dress, shoes, bag, hat, belt, scarf, sunglasses, or skin/hair/background (discarded). This is
called **semantic segmentation**. Each clothing region is cut out as a **transparent PNG** (the alpha
channel marks the fabric pixels), so later steps see only the garment.
`src/bg_removal.py` handles the cut-out, with **GrabCut** as a fallback for small accessories.

> One garment per region: a jacket worn over a t-shirt comes out as **one** upper-body garment.

### 2. Name each garment precisely
SegFormer only says "upper clothes". **CLIP** compares the cut-out with text prompts ("a t-shirt",
"a sweater", "a jacket"…) and picks the best match. This is **zero-shot classification**: no extra
training needed. It produces 16 specific types (t_shirt, sweater, shirt_blouse, jacket, top, pants,
leggings, shorts, skirt, dress, shoe, bag, hat, belt, scarf, sunglasses), each with a **broad category**
(top / bottom / dress / accessory). Names are kept tidy by `src/category_mapping.py`.

### 3. Describe and fingerprint each garment
- **Embedding (the fingerprint):** CLIP ViT-B/32 turns the cut-out into **512 numbers**, normalised to
  length 1 (**L2-normalised**). Garments that look alike get similar numbers.
- **Attributes (the description)** from `src/attribute_analyzer.py`:
  - *measured from pixels:* dominant **colour** (RGB + **HSV**, via k-means over fabric pixels only)
    and **silhouette** (fitted / boxy / A-line / tapered, from the mask's shape);
  - *guessed by CLIP zero-shot (noisier):* pattern, fabric, fit, formality, and a 0–100 formality score.
- **Outfit-level "fashion-theory" features:** hue distance, volume balance, formality coherence.

Text prompts are encoded once and cached, so attributes cost ~34 ms per garment on CPU and almost
nothing on GPU. If CLIP isn't available, colour and silhouette are still filled in.

## Why SegFormer

Real photos are messy: selfies, mirror shots, odd angles, dark clothes. Earlier approaches failed on them:

| Approach | What went wrong |
|---|---|
| Detectors trained on catalogue photos (DeepFashion2 / YOLOS) | Found "pants" on chests in mirror selfies; returned boxes with the room attached; produced duplicate boxes on one top (`sweater` + `shirt_blouse` + `t_shirt`) |
| Saliency cut-out (U2Net) | Treated the face as the subject and erased ~85% of a black t-shirt |
| GrabCut alone | Jagged edges that chopped off prints and hems |
| **SegFormer + CLIP (current)** | A human-parsing model trained on real poses keeps the whole garment. SegFormer picks the body region, then CLIP names the type on the clean cut-out. Embeddings contain only fabric, no background |

## What it saves (the output contract)

```
outputs/<id>/
  <garment>_<n>.png      transparent cut-out (RGBA)
  <garment>_<n>.npy      512-D CLIP embedding (float32, unit length)
  metadata.json          a JSON list, one entry per garment
  outfit_features.json   outfit-level hue / volume / formality numbers
  <id>_vis.jpg           annotated preview
```

One `metadata.json` entry:
```json
{
  "file": "t_shirt_0.png", "category": "t_shirt", "broad_category": "top",
  "raw_category": "Upper-clothes", "confidence": 0.82, "pixel_count": 137004,
  "bbox": [0, 291, 468, 702], "embedding_dim": 512,
  "attributes": {
    "color": {"dominant_rgb": [233, 196, 203], "hue": 348.8, "saturation": 0.158, "value": 0.915, "name": "red"},
    "silhouette": "boxy", "aspect_ratio": 1.165, "extent": 0.772, "area_px": 53097,
    "pattern": "dotted", "pattern_conf": 0.43, "fabric": "silk", "fabric_conf": 0.58,
    "fit": "regular", "fit_conf": 0.52, "formality": "formal", "formality_conf": 0.65,
    "formality_score": 72.6
  }
}
```

`outfit_features.json`:
```json
{"hue_contrast": 21.7, "top_bottom_hue_distance": 3.3, "volume_balance": 23.8,
 "top_bottom_area_ratio": 0.238, "formality_coherence": 79.9, "num_garments": 3}
```
Other modules depend on this exact layout. Don't change it without updating them.

## Run it

```bash
pip install -r config/requirements.txt
python src/run_pipeline.py --image ../samples/image2.jpg --output_dir outputs --device cuda
python src/run_pipeline.py --input_dir path/to/photos --output_dir outputs   # a whole folder
python src/render_perception_demo.py --outfit_dir outputs/<id> --photo ../samples/image2.jpg --out perception_demo.png
python tests/test_pipeline.py                                                   # checks
```
The easiest way to see it is the repo-root web demo: `python demo.py`, then choose **Module 1**.

## Files

| File | Role |
|---|---|
| `src/run_pipeline.py` | Entry point: one image or a folder → `outputs/<id>/` |
| `src/subject_check.py` | Person / animal check before parsing (YOLO11m-seg) |
| `src/fashion_segmenter.py` | SegFormer segmentation + CLIP subtype classification |
| `src/bg_removal.py` | Transparent cut-outs (mask-based, GrabCut fallback) |
| `src/clip_extract_embeddings.py` | 512-D CLIP embeddings |
| `src/attribute_analyzer.py` | Colour, silhouette, pattern, fabric, fit, formality + outfit features |
| `src/category_mapping.py` | Category names and broad categories |
| `src/render_perception_demo.py` | Slide figure: photo → garments + attributes |
| `tests/test_pipeline.py` | Checks (512-D unit-length embeddings, colour ignores background, …) |
| `tests/test_subject_check.py` | Person / animal rules, pet masking, pipeline stop; real photos when present |
| `scripts/eval_weak_labels.py` | Recall per garment type on Fashion144k, file-name words as weak labels |
| `src/yolo_detect_and_crop.py` | Older YOLO-based detector, kept for reference (not the default) |

**Note:** `Module3/` keeps identical copies of `src/fashion_segmenter.py`, `src/attribute_analyzer.py` and
`src/clip_extract_embeddings.py`. `Module3/tests/test_module1_sync.py` fails if they drift, so change both together.

## Accuracy (weak labels)
No labelled test set exists for Fashion144k, so `scripts/eval_weak_labels.py` uses the poster's garment words
in each file name as weak labels: recall = share of photos naming an item where Module 1 found it.
A lower bound (titles name items out of frame or hidden), recall only (titles omit items). All
144,169 photos, 2026-11-22; n = photos whose title names the item:

| Group | n | Recall | Exact type (where it differs) |
|---|---|---|---|
| Upper body | 68,774 | **97.6%** | jacket 74%, sweater 48%, shirt/blouse 30%, t-shirt 22%, top 14% |
| Lower body | 47,203 | **94.8%** | skirt 86%, pants 75%, leggings 42%, **shorts 4%** |
| Bag | 23,714 | 83.8% | |
| Hat | 10,823 | 79.5% | |
| Dress | 24,695 | 76.6% | |
| Shoes | 48,042 | 75.0% | |
| Scarf | 3,304 | 62.5% | |
| Belt | 3,127 | 19.4% | |
| Sunglasses | 4,861 | 5.6% | |

95% Wilson intervals are within ±1 point for every group except belt (±1.4) and scarf (±1.7).
Reading it:
- **Garment regions are found reliably** (upper 98%, lower 95%). The upper-body subtype is weaker:
  one region per body part means a title naming a jacket and a shirt can only match one of them, and
  the CLIP subtype often says jacket (shirt/blouse titles: 7,751 of the misses are "jacket").
- **Shorts are almost always called pants** (5,969 of 9,149 misses): Module 1 outputs 844 shorts in
  480,079 garments. A known CLIP zero-shot weakness; swap groups treat them alike, so Module 4 still
  offers the right group.
- **Small accessories are often missed** at Fashion144k's 400-px resolution: belts 19%, sunglasses 6%
  (often pushed up on the head, which SegFormer leaves as hair).
- Tights / stockings (no Module 1 type) come out as pants or leggings in 87% of 3,855 photos.

## Known limits
- One garment per body region, so layering isn't represented.
- CLIP's zero-shot guesses (pattern, fabric, fit) are noisy. For example, linen pants were labelled
  "corduroy" (confidence 0.34). Module 2 drops them (see [`docs/gnn-training.md`](../docs/gnn-training.md)).
  Colour, silhouette and the outfit-level features are reliable.
- Belts and small accessories aren't always split out as separate items.
- SegFormer needs a visible person: a crop showing only legs and boots comes out as background, so
  such photos give "no garments".
- Two people in one photo are parsed as one outfit.
