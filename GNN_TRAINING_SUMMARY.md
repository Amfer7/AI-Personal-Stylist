# GNN training summary

## Training approach

Module 2 builds one garment graph per outfit and trains a GNN to rank real Fashion144K outfits above outfits with one same-category garment swapped. It uses fashionability-weighted BPR loss and selects the checkpoint with the best validation AUC. See [training code](Module2/03_train_gnn.py) and [graph construction](Module2/02_outfit_dataset.py).

## Challenges and optimizations

| Challenge | Resolution |
| --- | --- |
| Reopening garment images and recomputing colour histograms slowed every epoch. | A [memory-mapped feature store](Module2/build_feature_store.py) computes these once. Training reads cached rows. |
| One outfit per GPU step added overhead. | Disconnected graphs are batched, with vectorized scoring and BPR loss. The [project notes](Module2/miniREADME.md) report a 5,000-outfit, 10-epoch run improving from about 300 seconds to 14 seconds, with unchanged AUC in a parity check. |
| Full-scale negative sampling scanned a large category pool for every swap, leaving GPU utilization around 3% and epochs around 27 minutes. | [Rejection sampling](Module2/03_train_gnn.py) picks a random same-category garment and retries a same-outfit match. The notes report about 1 minute per epoch afterward, roughly 27 times faster. |
| Blanket fp16 autocast caused a CLIP float32/float16 mismatch and repeated image-processing failures. | Autocast was restricted to SegFormer. A [consecutive-failure breaker](Module2/01_segment_batch.py) now stops systemic failures from marking thousands of images as processed. |
| Earlier training and inference used mismatched garment representations. | The current [CLIPOutfitGNN](Module2/03_train_gnn.py) consumes actual CLIP vectors directly, without a vocabulary lookup. |
| Noisy attributes and seed variation made model comparisons unreliable. | The team compared full attributes, no attributes, and an 11-dimensional subset across seeds 42, 43, and 44. The subset drops noisy pattern, fabric, and fit tags. Best-validation checkpoints were evaluated on the official test split. |

The recorded mean test AUCs were **0.8010** for the attribute subset, **0.7923** for the CLIP-and-colour baseline, and **0.7860** for full attributes. With three seeds, the differences were **not statistically significant**. The subset was chosen for its highest recorded mean and retained interpretable attributes and edge features. See [results](Module2/miniREADME.md) and a [local run log](Fashion144k_v1/ck_attr_subset_s42_log.txt).

## Hallucinations and misleading outputs

- **Earlier garment detection:** The [pipeline handoff guide](Module1/PIPELINE_HANDOFF_GUIDE.md) documents pants detected on chests in mirror selfies and duplicate garment labels for one top. SegFormer region parsing followed by CLIP classification of isolated crops replaced that approach.
- **Observed attribute error:** CLIP labelled linen pants as “corduroy” at 0.34 confidence in a real-photo check. This is noisy GNN input, not a GNN-generated garment. See [Module 1 verification](Module1/changes.md).
- **Potential false negatives:** A same-category garment swap may still yield a compatible outfit, but training labels every such swap as negative. The [project notes](Module2/miniREADME.md) flag this issue without measuring its frequency.
- **Score meaning:** The GNN produces a ranking score, not a calibrated probability. The [demo scorer](Module2/demo_score.py) presents it as a percentile relative to real test outfits.

The repository documents **no instance of the GNN itself hallucinating a garment or explanation**. The concrete errors above occur in the upstream perception and attribute pipeline.

## Results caveat

The [per-seed table](Module2/miniREADME.md) shows the subset beating full attributes on all three seeds. It does not show the baseline beating full attributes on every seed: on seed 44, full attributes scored **0.7885** versus **0.7677** for the baseline. All three variants remain a statistical tie based on the reported three-seed comparison.
