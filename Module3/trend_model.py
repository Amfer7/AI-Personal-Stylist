"""
trend_model.py

Standalone trendiness model for the Pinterest-based trend analysis module
(Module 3 in the project architecture). This file is self-contained and does
NOT depend on deepfashion2_parser.py, clip_extract_embeddings.py, the YOLOS
detection code, or the GNN compatibility code. It only assumes:

  - Garment / Pinterest embeddings are 512-D CLIP ViT-B/32 vectors, L2-normalized
    (same convention already used elsewhere in the repo).
  - Pinterest metadata has, at minimum, an image path and a year-month string
    (matches the "date_ym" column already shown in your scraping metadata:
    image_path, pin_id, date_ym, date_raw, board, query, description, ...).

What this gives you:
  1. encode_images()      -> CLIP-encode a folder of images
  2. fit()                -> cluster embeddings into style archetypes,
                              build a monthly popularity curve per cluster,
                              compute a growth + recency signal per cluster,
                              turn that into a 0-100 Trendiness Score per cluster
  3. save()/load()        -> persist the fitted model (centroids + scores + curves)
  4. score_embedding()    -> given ANY garment embedding (e.g. from a user photo,
                              or from a teammate's CLIP output), return a
                              trendiness score by matching to the nearest cluster
  5. score_image()        -> convenience wrapper: image path -> score

Nothing here requires the real Pinterest dataset to run — see synthetic_demo.py
for a test against fake data. When the real scrape is ready, call
`build_from_directory()` with the image folder + metadata CSV and everything
downstream (fit, save, score) is unchanged.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------------------------------
# Config (tune these — they are the "decisions" flagged in the review deck)
# --------------------------------------------------------------------------

DEFAULT_N_CLUSTERS = 30          # style archetypes to discover
GROWTH_WINDOW_MONTHS = 3         # "recent" window size for growth calc
RECENCY_HALF_LIFE_MONTHS = 4     # how fast older months decay in weight
MIN_CLUSTER_IMAGES = 15          # clusters with fewer images get capped/flagged
MATCH_CONFIDENCE_THRESHOLD = 0.20  # min cosine similarity to trust a match
GROWTH_WEIGHT = 0.6              # combining weights for final score
RECENCY_WEIGHT = 0.4


# --------------------------------------------------------------------------
# CLIP encoding
# --------------------------------------------------------------------------

class ClipEncoder:
    """
    Wraps Prateek's clip_extract_embeddings.py directly — does NOT load its
    own separate CLIP model. This guarantees every embedding in the project
    (garment crops AND Pinterest images) comes from the exact same model
    instance, same preprocessing pipeline, and same RGBA-compositing logic,
    with zero risk of drift between two independently-written encoders.

    Requires clip_extract_embeddings.py to be importable (same folder as
    this file, or added to sys.path before importing trend_model).
    """

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self._cee = None  # Prateek's clip_extract_embeddings module, lazy-imported
        self._model = None
        self._preprocess = None

    def _ensure_loaded(self):
        if self._cee is not None:
            return
        try:
            import clip_extract_embeddings as _cee
        except ImportError as e:
            raise ImportError(
                "Could not import clip_extract_embeddings.py (Prateek's "
                "embedding module). Put it in the same folder as "
                "trend_model.py, or add its folder to sys.path before "
                "importing trend_model."
            ) from e
        self._cee = _cee
        # get_clip_model() caches internally on the module side too, but we
        # call it once here to force-load and to grab the exact model +
        # preprocess instances for batched encoding below.
        self._model, self._preprocess, self.device = _cee.get_clip_model(
            model_name=self.model_name, device=self.device
        )

    def encode_images(self, image_paths: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """
        Returns an (N, 512) L2-normalized float32 array, same order as
        image_paths. Batches the forward pass for speed on large datasets,
        but uses Prateek's exact model instance, preprocess function, and
        RGBA-white-composite logic — so results are identical to calling
        his embed_image() one image at a time, just faster in bulk.
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        all_embeds = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                img_pil = Image.open(p)
                # Same RGBA-composite handling as clip_extract_embeddings.py's
                # embed_image(), so transparent garment crops don't pick up
                # black-boundary artifacts. No-op for plain RGB Pinterest JPEGs.
                if img_pil.mode == "RGBA":
                    white_bg = Image.new("RGBA", img_pil.size, (255, 255, 255, 255))
                    img_rgb = Image.alpha_composite(white_bg, img_pil).convert("RGB")
                else:
                    img_rgb = img_pil.convert("RGB")
                imgs.append(self._preprocess(img_rgb))
            batch = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeds, axis=0)

    def encode_images_unbatched(self, image_paths: Sequence[str]) -> np.ndarray:
        """
        Alternative to encode_images(): calls Prateek's embed_image() directly,
        one image at a time, with zero reimplementation on this side. Use this
        instead of encode_images() if you want the literal function call
        reused rather than a batched reimplementation of its logic — slower
        on large datasets (no batching), but there is no code here that
        duplicates his preprocessing at all.
        """
        self._ensure_loaded()
        embeds = [self._cee.embed_image(p, device=self.device) for p in image_paths]
        return np.stack(embeds).astype(np.float32)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------
# Data container for a fitted model
# --------------------------------------------------------------------------

@dataclass
class ClusterInfo:
    cluster_id: int
    centroid: np.ndarray                  # (512,) L2-normalized
    n_images: int
    monthly_counts: Dict[str, int]         # "YYYY-MM" -> count
    growth_raw: float
    recency_raw: float
    trendiness_score: float                # 0-100, final output
    low_confidence: bool                   # True if n_images < MIN_CLUSTER_IMAGES


@dataclass
class TrendModel:
    n_clusters: int = DEFAULT_N_CLUSTERS
    clusters: Dict[int, ClusterInfo] = field(default_factory=dict)
    kmeans: Optional[KMeans] = None
    fitted_on: Optional[str] = None  # timestamp string

    # -- fitting ------------------------------------------------------

    def fit(
        self,
        embeddings: np.ndarray,
        date_ym: Sequence[str],
        n_clusters: Optional[int] = None,
    ) -> "TrendModel":
        """
        embeddings: (N, 512) L2-normalized CLIP embeddings
        date_ym:    list/array of "YYYY-MM" strings, same order as embeddings
        """
        assert len(embeddings) == len(date_ym), "embeddings and date_ym must align"
        k = n_clusters or self.n_clusters
        self.n_clusters = k

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(embeddings)
        self.kmeans = km

        clusters: Dict[int, ClusterInfo] = {}
        for cid in range(k):
            mask = labels == cid
            n_images = int(mask.sum())
            months = [date_ym[i] for i in range(len(date_ym)) if mask[i]]
            monthly_counts = _monthly_counts(months)

            growth_raw = _growth_score(monthly_counts)
            recency_raw = _recency_score(monthly_counts)

            centroid = km.cluster_centers_[cid]
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            clusters[cid] = ClusterInfo(
                cluster_id=cid,
                centroid=centroid.astype(np.float32),
                n_images=n_images,
                monthly_counts=monthly_counts,
                growth_raw=growth_raw,
                recency_raw=recency_raw,
                trendiness_score=0.0,  # filled in below after normalization
                low_confidence=n_images < MIN_CLUSTER_IMAGES,
            )

        _assign_trendiness_scores(clusters)
        self.clusters = clusters
        self.fitted_on = datetime.utcnow().isoformat()
        return self

    # -- quality check --------------------------------------------------

    def suggest_k(self, embeddings: np.ndarray, k_range: Sequence[int]) -> Dict[int, float]:
        """Silhouette score for a range of k values, to help pick n_clusters.
        Run this once during development, not as part of the production pipeline."""
        scores = {}
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(embeddings)
            scores[k] = float(silhouette_score(embeddings, km.labels_))
        return scores

    # -- scoring new garments --------------------------------------------

    def score_embedding(self, embedding: np.ndarray) -> Dict:
        """
        embedding: (512,) L2-normalized vector (garment or user-photo crop,
        must be from the SAME CLIP model / same vector space).

        Returns a dict with the score, matched cluster, and a confidence flag.
        If nothing in the trend model looks like this garment, returns
        trendiness_score=None rather than a fabricated number.
        """
        if not self.clusters:
            raise RuntimeError("Model not fitted yet — call fit() first.")

        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        centroids = np.stack([c.centroid for c in self.clusters.values()])
        cluster_ids = list(self.clusters.keys())

        sims = centroids @ embedding  # cosine similarity (both L2-normalized)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_cluster = self.clusters[cluster_ids[best_idx]]

        if best_sim < MATCH_CONFIDENCE_THRESHOLD:
            return {
                "trendiness_score": None,
                "reason": "no confident match to any known style cluster",
                "best_similarity": best_sim,
                "cluster_id": None,
            }

        return {
            "trendiness_score": best_cluster.trendiness_score,
            "cluster_id": best_cluster.cluster_id,
            "similarity": best_sim,
            "cluster_size": best_cluster.n_images,
            "low_confidence_cluster": best_cluster.low_confidence,
        }

    def score_image(self, image_path: str, encoder: Optional[ClipEncoder] = None) -> Dict:
        encoder = encoder or ClipEncoder()
        emb = encoder.encode_images([image_path])[0]
        return self.score_embedding(emb)

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "TrendModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> pd.DataFrame:
        rows = []
        for c in sorted(self.clusters.values(), key=lambda c: -c.trendiness_score):
            rows.append({
                "cluster_id": c.cluster_id,
                "n_images": c.n_images,
                "trendiness_score": round(c.trendiness_score, 1),
                "growth_raw": round(c.growth_raw, 3),
                "recency_raw": round(c.recency_raw, 3),
                "low_confidence": c.low_confidence,
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Temporal scoring helpers
# --------------------------------------------------------------------------

def _monthly_counts(months: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in months:
        counts[m] = counts.get(m, 0) + 1
    return dict(sorted(counts.items()))


def _sorted_months(monthly_counts: Dict[str, int]) -> List[str]:
    return sorted(monthly_counts.keys())


def _growth_score(monthly_counts: Dict[str, int]) -> float:
    """% change: mean of last GROWTH_WINDOW_MONTHS vs mean of the
    GROWTH_WINDOW_MONTHS before that. Returns 0.0 for clusters with too
    little history to say anything meaningful (avoids noisy spikes from
    tiny clusters dominating the ranking)."""
    months = _sorted_months(monthly_counts)
    if len(months) < GROWTH_WINDOW_MONTHS * 2:
        return 0.0

    recent = months[-GROWTH_WINDOW_MONTHS:]
    prior = months[-2 * GROWTH_WINDOW_MONTHS : -GROWTH_WINDOW_MONTHS]

    recent_mean = np.mean([monthly_counts[m] for m in recent])
    prior_mean = np.mean([monthly_counts[m] for m in prior])

    if prior_mean < 1e-6:
        return float(recent_mean > 0)  # went from nothing to something = growth signal, capped

    return float((recent_mean - prior_mean) / prior_mean)


def _recency_score(monthly_counts: Dict[str, int]) -> float:
    """Exponentially recency-weighted share of a cluster's images. A cluster
    whose images skew toward the most recent months scores higher than one
    with the same total count spread evenly or skewed old."""
    months = _sorted_months(monthly_counts)
    if not months:
        return 0.0

    latest = _parse_ym(months[-1])
    total = sum(monthly_counts.values())
    if total == 0:
        return 0.0

    weighted = 0.0
    for m, count in monthly_counts.items():
        age_months = _month_diff(latest, _parse_ym(m))
        weight = 0.5 ** (age_months / RECENCY_HALF_LIFE_MONTHS)
        weighted += count * weight

    return weighted / total  # in (0, 1], higher = more recent-skewed


def _parse_ym(ym: str) -> Tuple[int, int]:
    year, month = ym.split("-")
    return int(year), int(month)


def _month_diff(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return (a[0] - b[0]) * 12 + (a[1] - b[1])


def _assign_trendiness_scores(clusters: Dict[int, ClusterInfo]) -> None:
    """Normalize growth + recency across all clusters (percentile rank, so
    the score is relative to what's actually in this batch of Pinterest data)
    and combine into a single 0-100 Trendiness Score."""
    ids = list(clusters.keys())
    growth_vals = np.array([clusters[i].growth_raw for i in ids])
    recency_vals = np.array([clusters[i].recency_raw for i in ids])

    growth_pct = _percentile_rank(growth_vals)
    recency_pct = _percentile_rank(recency_vals)

    for idx, cid in enumerate(ids):
        combined = GROWTH_WEIGHT * growth_pct[idx] + RECENCY_WEIGHT * recency_pct[idx]
        score = float(combined * 100)
        if clusters[cid].low_confidence:
            # don't let a sparse cluster claim a top score off noise
            score = min(score, 50.0)
        clusters[cid].trendiness_score = round(score, 2)


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort().argsort()
    n = len(values)
    if n <= 1:
        return np.zeros_like(values, dtype=float)
    return order / (n - 1)


# --------------------------------------------------------------------------
# End-to-end helper for when the real Pinterest data arrives
# --------------------------------------------------------------------------

def build_from_directory(
    metadata_csv: str,
    image_col: str = "image_path",
    date_col: str = "date_ym",
    n_clusters: int = DEFAULT_N_CLUSTERS,
    encoder: Optional[ClipEncoder] = None,
    embedding_cache_path: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> TrendModel:
    """
    metadata_csv: CSV matching your scraping metadata schema, needs at least
                  image_path and date_ym columns (matches your build script's
                  output: image_path, pin_id, date_ym, date_raw, board,
                  query, description, orig_width, orig_height, hash).

    base_dir: folder that `image_path` values are relative to. Defaults to
              the folder metadata_csv lives in — matches your dataset layout
              (pinterest_dataset/metadata.csv with image_path values like
              "images/2533343533223371.jpg"). Pass this explicitly if you
              move the CSV somewhere else without moving the images.

    This is the ONE function you'll call once the real scrape is ready.
    Everything else (fit, save, score) already works today against
    synthetic data — see synthetic_demo.py.
    """
    df = pd.read_csv(metadata_csv)
    if image_col not in df.columns or date_col not in df.columns:
        raise ValueError(f"metadata_csv must have '{image_col}' and '{date_col}' columns")

    df = df.dropna(subset=[image_col, date_col]).reset_index(drop=True)

    base = base_dir or os.path.dirname(os.path.abspath(metadata_csv))
    # Windows-built CSVs store image_path with backslashes (e.g. "images\x.jpg").
    # Normalize to forward slashes so this works on Linux (Colab) too.
    normalized = [str(p).replace("\\", "/") for p in df[image_col]]
    resolved_paths = [
        p if os.path.isabs(p) else os.path.join(base, p) for p in normalized
    ]
    missing = [p for p in resolved_paths[:50] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the first 50 image paths don't exist on disk "
            f"(e.g. {missing[0]}). Check that base_dir is correct — it "
            f"defaults to the metadata_csv's own folder."
        )
    df[image_col] = resolved_paths

    if embedding_cache_path and os.path.exists(embedding_cache_path):
        embeddings = np.load(embedding_cache_path)
    else:
        encoder = encoder or ClipEncoder()
        embeddings = encoder.encode_images(df[image_col].tolist())
        if embedding_cache_path:
            np.save(embedding_cache_path, embeddings)

    model = TrendModel(n_clusters=n_clusters)
    model.fit(embeddings, df[date_col].tolist())
    return model


# --------------------------------------------------------------------------
# Per-garment / per-category extension
#
# Everything above scores a WHOLE PHOTO as one style. This section instead
# fits an INDEPENDENT TrendModel per garment category (top, bottom, dress,
# shoe, accessory, ...), so "which tops are trending" and "which bottoms are
# trending" are answered separately rather than mixed into one clustering
# space. It reuses TrendModel unmodified (all the tested clustering/growth/
# recency/matching logic) — just runs one instance per category.
# --------------------------------------------------------------------------

DEFAULT_N_CLUSTERS_PER_CATEGORY = 15  # smaller than the whole-photo default (30),
                                       # since each category has fewer garments than
                                       # the full dataset has photos

# Per-category k overrides, chosen with k_sweep.py. At k=15, dresses were only
# moderately stable (ARI 0.58); k=8 gives ARI 0.94 (k=10: 0.84, k=12: 0.89).
N_CLUSTERS_OVERRIDES: Dict[str, int] = {"dress": 8}


def k_for_category(category: str, n_images: int,
                   n_clusters_per_category: int = DEFAULT_N_CLUSTERS_PER_CATEGORY,
                   overrides: Optional[Dict[str, int]] = None) -> int:
    """The single k rule shared by fitting, the stability test and the
    temporal backtest: the per-category override if any, else the default —
    capped so we don't ask for more clusters than there are (roughly) images
    to fill them."""
    overrides = N_CLUSTERS_OVERRIDES if overrides is None else overrides
    k = overrides.get(category, n_clusters_per_category)
    return min(k, max(2, n_images // 20))


@dataclass
class PerCategoryTrendModel:
    models: Dict[str, TrendModel] = field(default_factory=dict)
    fitted_on: Optional[str] = None

    def fit(
        self,
        embeddings: np.ndarray,
        categories: Sequence[str],
        date_ym: Sequence[str],
        n_clusters_per_category: int = DEFAULT_N_CLUSTERS_PER_CATEGORY,
        min_images_per_category: int = 100,
    ) -> "PerCategoryTrendModel":
        assert len(embeddings) == len(categories) == len(date_ym), \
            "embeddings, categories, and date_ym must all align"

        categories = np.array(categories)
        unique_categories = sorted(set(map(str, categories)))
        models: Dict[str, TrendModel] = {}

        for cat in unique_categories:
            mask = categories == cat
            n_images = int(mask.sum())
            if n_images < min_images_per_category:
                print(f"Skipping category '{cat}': only {n_images} images "
                      f"(< {min_images_per_category} minimum) — too little data "
                      f"to cluster meaningfully.")
                continue

            k = k_for_category(cat, n_images, n_clusters_per_category)
            print(f"Fitting category '{cat}': {n_images} images, k={k}")
            model = TrendModel(n_clusters=k)
            model.fit(embeddings[mask], [date_ym[i] for i in range(len(date_ym)) if mask[i]])
            models[cat] = model

        self.models = models
        self.fitted_on = datetime.utcnow().isoformat()
        return self

    def score_embedding(self, embedding: np.ndarray, category: str) -> Dict:
        """Score a single garment. `category` must match one of the broad
        categories this model was fit on (e.g. 'top', 'bottom', 'dress',
        'shoe', 'accessory') — typically fashion_segmenter.py's
        `broad_category` field for the garment."""
        if category not in self.models:
            return {
                "trendiness_score": None,
                "reason": f"no trend model fitted for category '{category}' "
                          f"(known categories: {sorted(self.models.keys())})",
                "cluster_id": None,
            }
        result = self.models[category].score_embedding(embedding)
        result["category"] = category
        return result

    def score_outfit(self, garments: List[Dict]) -> Dict:
        """
        garments: list of {"embedding": np.ndarray, "category": str, ...}
        (any extra keys, e.g. a filename or bbox, are passed through untouched).

        Returns per-garment scores plus which garment has the lowest
        trendiness ("weak link") — the trend-model analogue of the GNN's
        weak-link identification for harmony.
        """
        scored = []
        for g in garments:
            result = self.score_embedding(g["embedding"], g["category"])
            scored.append({**g, **result})

        scorable = [g for g in scored if g["trendiness_score"] is not None]
        weak_link = min(scorable, key=lambda g: g["trendiness_score"]) if scorable else None

        return {
            "garments": scored,
            "weak_link_category": weak_link["category"] if weak_link else None,
            "weak_link_score": weak_link["trendiness_score"] if weak_link else None,
            "outfit_avg_trendiness": (
                float(np.mean([g["trendiness_score"] for g in scorable])) if scorable else None
            ),
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PerCategoryTrendModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def summary(self) -> pd.DataFrame:
        rows = []
        for cat, model in self.models.items():
            for _, row in model.summary().iterrows():
                rows.append({"category": cat, **row.to_dict()})
        return pd.DataFrame(rows)


def fragment_mask(
    manifest: pd.DataFrame,
    category: str = "dress",
    dominant: str = "top",
    category_col: str = "broad_category",
) -> np.ndarray:
    """
    Boolean mask (aligned with `manifest` rows) of likely SegFormer fragments:
    a `category` mask that is SMALLER than a `dominant` mask in the same source
    photo. On the Pinterest scrape ~half of all 'dress' detections co-occur
    with a 'top', and in ~4k of those the dress mask is the smaller one —
    typically the hem of a long top or a jacket split into two labels. These
    crops are not real dresses and blur the dress clusters.

    Fixing this upstream would mean editing the vendored fashion_segmenter.py
    (guarded by test_module1_sync.py), so it is filtered here instead.
    """
    px = manifest.groupby(["source_idx", category_col])["pixel_count"].sum().unstack(fill_value=0)
    dominant_px = manifest["source_idx"].map(px[dominant]) if dominant in px else 0
    return ((manifest[category_col] == category) & (manifest["pixel_count"] < dominant_px)).values


def build_per_category_from_manifest(
    manifest_csv: str,
    embeddings_path: str,
    category_col: str = "broad_category",
    date_col: str = "date_ym",
    n_clusters_per_category: int = DEFAULT_N_CLUSTERS_PER_CATEGORY,
    end_month: Optional[str] = None,
    drop_fragments: bool = False,
) -> PerCategoryTrendModel:
    """
    The per-garment equivalent of build_from_directory(). Reads the manifest
    + embeddings produced by pinterest_garment_segmentation.py and fits one
    TrendModel per garment category.

    end_month:      drop garments dated after this "YYYY-MM". Use it to cut a
                    partial final month (the scrape ended 2026-09-11), which
                    would otherwise sit inside the growth/recency windows.
    drop_fragments: drop likely SegFormer fragments (see fragment_mask()).
    """
    manifest = pd.read_csv(manifest_csv)
    embeddings = np.load(embeddings_path)
    assert len(manifest) == len(embeddings), (
        f"Row count mismatch: manifest has {len(manifest)} rows, embeddings "
        f"has {len(embeddings)}. Are these from the same segmentation run?"
    )
    keep = np.ones(len(manifest), dtype=bool)
    if end_month:
        keep &= (manifest[date_col] <= end_month).values
    if drop_fragments:
        keep &= ~fragment_mask(manifest, category_col=category_col)
    if not keep.all():
        print(f"Filtering: keeping {int(keep.sum())} of {len(manifest)} garments")
        manifest, embeddings = manifest[keep].reset_index(drop=True), embeddings[keep]

    model = PerCategoryTrendModel()
    model.fit(
        embeddings,
        manifest[category_col].tolist(),
        manifest[date_col].tolist(),
        n_clusters_per_category=n_clusters_per_category,
    )
    return model
