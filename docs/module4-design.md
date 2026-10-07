# Module 4: Swap Recommendation (design)

**Status:** designed, not built yet · all decisions made 2026-10-07.
**Full engineering spec** (every decision with alternatives, evidence and file references):
[`module4-design-detailed.md`](module4-design-detailed.md).

---

## What it does

You give it one outfit photo. It answers:

> **"Which one piece should I change, and what should I change it to?"**

It returns the piece to swap, three replacement suggestions (with pictures), and how each one
changes the outfit's **harmony** (do the pieces go together?) and **trend** (is it popular right now?).
If the outfit is already good, it says so instead of inventing a change.

## How it fits with the other modules

```
 photo
   │
   ▼
 M1  "What's in the photo?"              → cuts out each garment and describes it
   │
   ├──► M2  "Do these pieces go together?"   → harmony score
   ├──► M3  "Is each piece popular now?"      → trend score
   ▼
 M4  "What one swap improves this most?"  → tries replacements, scores each with M2 + M3
```

| Module | Takes in | Produces | Used by M4 for |
|---|---|---|---|
| **M1 Perception** | a photo | one cut-out per garment + a 512-number **CLIP embedding** (a visual "fingerprint") + attributes (colour, formality, silhouette…) | describing your current outfit |
| **M2 Harmony GNN** | the full garment descriptions of an outfit | one **harmony score** for the whole outfit | judging every "what-if" outfit |
| **M3 Trend model** | a garment's embedding + category | a **trend score** per garment (from Pinterest style clusters) | judging how current a replacement is |
| **Feature store** | M1 run offline over whole datasets | a catalogue of garments with full descriptions | the pool of possible replacements |

**Key constraint:** M2 scores whole outfits only. It can't say "the shoes are the problem". So M4
finds the weak piece by **trying swaps**: replace a piece, re-score the outfit, see what improves.

---

## The decisions, in plain terms

### 1. Where replacements come from
A combined catalogue of **Fashion144k** (~480k garments) plus **Pinterest** (~144k garments,
re-processed so M2 can score them). Pinterest is the only *current* source and the one M3 measures
trend on.
- **Evaluation** uses only garments M2 never trained on (the *test split*), so the results are honest.
- **The demo** uses everything, for the most variety.

### 2. Which piece to swap
For every piece, M4 tries every allowed replacement and measures the improvement. It then summarises
each piece with one number and swaps the piece that improves most.
- Taking only the *single best* replacement is misleading. Out of thousands of tries, one will look
  great by luck (the **winner's curse**). So the default is the **average of the top 5**.
- Three summary rules are compared in evaluation and the winner ships.

### 3. How harmony and trend combine
`gain = w_H × (harmony change) + w_T × (trend change)`, with both changes put on the same scale
(divided by their typical spread) so the weights mean what they say.
- The harmony change uses the GNN's **raw score**, not the 0–100 percentile, which caps at 100 and
  makes good outfits impossible to compare.
- The trend change is measured **for the swapped piece**, so it doesn't get watered down in big outfits.
- The weight is chosen from a **trade-off curve**: as much trend as possible while keeping ≥ 90% of the
  harmony improvement. A slider in the demo lets you explore it.

### 4. What may replace what
M1 finds **one garment per body region**, so each outfit has *slots*:

| Slot | Can be replaced by |
|---|---|
| Upper body | t-shirt, sweater, shirt/blouse, jacket, top |
| Lower body | pants, leggings, shorts, skirt |
| Dress | dress |
| Shoe, bag, hat, belt, scarf, sunglasses | the same type only |

A **co-occurrence filter** learned from real outfits blocks combinations that never happen
(e.g. pants + leggings). Whether a swap *looks good* is always the GNN's call.

### 5. How many candidates are checked
**All of them.** Instead of sampling, M4 scores every allowed replacement. That's made fast by building
all the "what-if" outfits in one GPU operation and precomputing part of the GNN for the whole
catalogue. Target: **≤ 5 seconds per photo** on the main machine, using at most **6 GB of GPU memory**.

### 6. What you see
Three possible verdicts:

| Verdict | Message |
|---|---|
| **swap** | "Swap your shoes": top 3 replacements, plus other pieces that are also worth improving |
| **refresh** | "Already strong. Optional: these are trendier and keep the look" |
| **keep** | "Already strong. No swap recommended" |

Each suggestion shows the picture, any type change ("jacket → shirt"), harmony and trend before → after,
where it comes from, and a short **"what changed"** line (e.g. "formality now closer to your jeans").
That line *describes* the difference; it doesn't claim to know the model's reasoning.

### 7. When the outfit is already good
Searching thousands of replacements always finds *some* tiny improvement by luck. So the bar for
recommending a swap is **learned from data**: run M4 on real outfits (should say "no swap") and on
deliberately spoiled ones (should say "swap"), and pick the threshold that separates them best.

### 8. No duplicates, real variety
- Identical garments in the catalogue (reposts) are merged once, up front.
- M4 never suggests the item you're already wearing.
- The three suggestions come from **different style clusters**, so they're real choices, not three
  versions of the same black boot.

### 9. How we prove it works
1. Take real outfits, **spoil one piece**, and check whether M4 finds it.
2. Check whether M4 ranks the **garment the person actually wore** near the top. This is the
   strongest evidence, because it's a real human choice rather than the model grading itself.
3. Compare against simple baselines (random swap, trend-only, a CLIP-similarity heuristic).
4. A **blind human study**: volunteer raters pick the better outfit (M4's fix vs the spoiled one, and vs a
   random fix).
5. All tuning happens on one set of outfits; the final numbers come from a separate set, run once.

*(A vision-language-model judge is under consideration.)*

### 10. The demo
Module 4 is added to the existing `demo.py` web app: a verdict headline, suggestion cards, a trend-weight
slider that updates instantly, and a **"Try this swap"** button showing an outfit board with the new
piece.

### 11. The trend formula
M3's backtest showed that "growth" doesn't predict the future; recent surges tend to fall back.
Because M4 *acts* on the trend score, the formula is re-tested before M4 is tuned. The new test asks
"is this style still widely seen months later?", and the simplest formula wins any tie.

---

## Build order

**Phase 0: Pinterest into the catalogue** (before M4)
1. Get the full-resolution Pinterest photos (the copies on disk are shrunk and squashed to 224×224).
2. Run M1 over them in a mode that saves everything M2 needs.
3. Split into train / validation / test.
4. Check that M2 still works on Pinterest outfits. Retrain only if it doesn't.
5. Refit M3 on the new data and settle the trend formula.
6. Build the combined catalogue.

**Then M4:** candidate pool → swap groups → recommender → evaluation → demo.

## Not in scope (for now)
- **Multi-piece swaps.** Deferred, but the code is designed so they can be added later.
- Personal preferences, size/price/availability, generating new garments.
