"""Candidate pool (D1, D5, D8a): every garment in the configured feature stores, with what Module 4
needs per row, built once and cached under <cache_dir>/pool/.

Per row: source, split, specific type, broad category, outfit, file, the GNN input projection
(precomputed, D5), M3 trend score + cluster (vectorised, D3/D8c), and near-duplicate
representatives for the demo pool and for the eval pool (D8a). Global row = store offset + store
row, so CLIP embeddings and node scalars are read straight from the stores at load time.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from m2_bridge import BROAD_ID, BROAD_VOCAB, FEATURE_COLS, TREND_MATCH_THRESHOLD, od
from swap_groups import TYPE_ID, TYPES

POOL_FORMAT = 2
SPLIT_CODES = {"train": 0, "val": 1, "test": 2, "none": 3}
_CHUNK = 65536


def pool_dir(settings) -> str:
    return settings.abs(os.path.join(settings.cache_dir, "pool"))


def _sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def pool_fingerprint(settings) -> dict:
    """Everything the cached pool depends on. A mismatch means: rebuild."""
    stores = []
    for s in settings.stores:
        feat = np.load(settings.abs(os.path.join(s.store_dir, "features.npy")), mmap_mode="r")
        root = settings.abs(s.output_root)
        crops = len(os.listdir(root)) if os.path.isdir(root) else 0     # outfit folders added or removed
        stores.append({"name": s.name, "rows": int(feat.shape[0]), "split": _sha16(settings.abs(s.split)),
                       "output_entries": crops})
    return {"format": POOL_FORMAT, "checkpoint": _sha16(settings.abs(settings.checkpoint)),
            "trend_model": _sha16(settings.abs(settings.trend_model)), "t_dup": settings.t_dup,
            "stores": stores}


def is_fresh(settings) -> bool:
    p = os.path.join(pool_dir(settings), "fingerprint.json")
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as f:
        return json.load(f) == pool_fingerprint(settings)


def load_store_split(settings, spec) -> dict:
    """outfit_key -> "train" | "val" | "test". Keys missing from the split count as "none"."""
    path = settings.abs(spec.split)
    if path.endswith(".mat"):
        s = od.load_split(path)
        return {str(int(i)): name for name, ids in s.items() for i in ids}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_store_arrays(settings):
    """CLIP embeddings [N, 512] and node scalars [N, 3] (hue|NaN, formality, area), pool row order."""
    embs, scals = [], []
    for s in settings.stores:
        d = settings.abs(s.store_dir)
        feat = np.load(os.path.join(d, "features.npy"), mmap_mode="r")
        embs.append(np.ascontiguousarray(feat[:, :512]))
        scals.append(np.load(os.path.join(d, "node_scalars.npy")))
    return np.concatenate(embs), np.concatenate(scals)


def crops_present(output_root: str, files_by_key: dict) -> dict:
    """{outfit_key: set of files} actually on disk. Only the demo pool needs crops (they are shown);
    scoring reads the feature store, so the eval pool does not."""
    if not os.path.isdir(output_root):
        return {}
    with os.scandir(output_root) as it:
        dirs = {e.name for e in it if e.is_dir()}
    return {k: set(os.listdir(os.path.join(output_root, k))) & set(fs)
            for k, fs in files_by_key.items() if k in dirs}


def _outfit_sort_key(k: str):
    return (0, int(k), "") if k.isdigit() else (1, 0, k)


def trend_scores(emb: np.ndarray, broad_names: np.ndarray, trend_model):
    """Vectorised PerCategoryTrendModel.score_embedding: one matmul per broad category.
    Returns (score float32 [N], NaN = no model / no confident match; cluster int16 [N], -1 = none)."""
    n = emb.shape[0]
    score = np.full(n, np.nan, np.float32)
    cluster = np.full(n, -1, np.int16)
    for cat, tm in trend_model.models.items():
        rows = np.nonzero(broad_names == cat)[0]
        if rows.size == 0:
            continue
        ids = np.asarray(list(tm.clusters.keys()))
        C = np.stack([tm.clusters[c].centroid for c in ids]).astype(np.float32)
        ts = np.array([tm.clusters[c].trendiness_score for c in ids], np.float32)
        for s in range(0, rows.size, _CHUNK):
            r = rows[s:s + _CHUNK]
            e = emb[r]
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
            sims = e @ C.T
            best = sims.argmax(axis=1)
            ok = sims[np.arange(r.size), best] >= TREND_MATCH_THRESHOLD
            score[r[ok]] = ts[best[ok]]
            cluster[r[ok]] = ids[best[ok]]
    return score, cluster


def dedupe(emb: torch.Tensor, type_ids: np.ndarray, area: np.ndarray, t_dup: float,
           eligible: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Near-duplicate groups (D8a): within each specific type, rows linked by cosine >= t_dup
    (transitively) share one representative, the largest crop (ties: lowest row).
    emb: [N, 512] L2-normalised, on the GPU when available. Returns rep_of int32 [N], -1 = not eligible."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rep = np.full(len(type_ids), -1, np.int32)
    for t in np.unique(type_ids[eligible]):
        rows = np.nonzero(eligible & (type_ids == t))[0]
        E = emb[torch.as_tensor(rows, device=emb.device)]
        src, dst = [], []
        for s in range(0, len(rows), chunk):
            ii, jj = torch.nonzero(E[s:s + chunk] @ E.T >= t_dup, as_tuple=True)
            ii = ii + s
            keep = jj > ii
            src.append(ii[keep].cpu().numpy())
            dst.append(jj[keep].cpu().numpy())
        src, dst, m = np.concatenate(src), np.concatenate(dst), len(rows)
        _, comp = connected_components(coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(m, m)),
                                       directed=False)
        order = np.lexsort((rows, -area[rows], comp))       # by component, then area desc, then row
        c = comp[order]
        first = np.ones(m, bool)
        first[1:] = c[1:] != c[:-1]
        rep_of_comp = np.empty(comp.max() + 1, np.int64)
        rep_of_comp[c[first]] = rows[order][first]
        rep[rows] = rep_of_comp[comp]
    return rep


def build_pool(settings, gnn, trend_model) -> None:
    """Build and write <cache_dir>/pool/ (everything except cooc.npy, reference.json, fingerprint)."""
    out = pool_dir(settings)
    os.makedirs(out, exist_ok=True)
    parts, files, outfit_keys, proj = [], [], {}, []
    for sid, spec in enumerate(settings.stores):
        d = settings.abs(spec.store_dir)
        with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        n = int(index["num_rows"])
        splits = load_store_split(settings, spec)
        keys = sorted(index["outfits"], key=_outfit_sort_key)
        outfit_keys[spec.name] = keys
        type_ = np.full(n, -1, np.int8)
        broad = np.full(n, -1, np.int8)
        split = np.full(n, SPLIT_CODES["none"], np.uint8)
        ordv = np.full(n, -1, np.int32)
        crop = np.zeros(n, bool)
        f_ = [None] * n
        on_disk = crops_present(settings.abs(spec.output_root),
                                {k: [e["file"] for e in es] for k, es in index["outfits"].items()})
        for o, key in enumerate(keys):
            sp = SPLIT_CODES[splits.get(key, "none")]
            for e in index["outfits"][key]:
                r = e["row"]
                type_[r] = TYPE_ID.get(e["category"], -1)
                broad[r] = BROAD_ID[e["broad_category"]]
                split[r] = sp
                ordv[r] = o
                f_[r] = e["file"]
                crop[r] = e["file"] in on_disk.get(key, ())
        assert (ordv >= 0).all(), f"{spec.name}: some store rows are not listed in index.json"
        files.extend(f_)
        parts.append({"source": np.full(n, sid, np.uint8), "split": split, "type": type_, "broad": broad,
                      "store_row": np.arange(n, dtype=np.int32), "outfit_ord": ordv, "has_crop": crop})
        feat = np.load(os.path.join(d, "features.npy"), mmap_mode="r")
        for s in range(0, n, _CHUNK):
            x = torch.from_numpy(np.ascontiguousarray(feat[s:s + _CHUNK])[:, FEATURE_COLS])
            proj.append(gnn.project(x).cpu().numpy())
    meta = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    emb, scal = load_store_arrays(settings)
    meta["trend"], meta["cluster"] = trend_scores(emb, np.asarray(BROAD_VOCAB)[meta["broad"]], trend_model)
    emb_t = torch.as_tensor(emb, device=gnn.device)
    emb_t = emb_t / emb_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    known = meta["type"] >= 0
    test = meta["split"] == SPLIT_CODES["test"]
    # Demo suggestions are shown as images, so the demo pool is limited to garments with a crop on disk.
    meta["rep_demo"] = dedupe(emb_t, meta["type"], scal[:, 2], settings.t_dup, eligible=known & meta["has_crop"])
    meta["rep_eval"] = dedupe(emb_t, meta["type"], scal[:, 2], settings.t_dup, eligible=known & test)
    del emb_t
    np.savez(os.path.join(out, "meta.npz"), **meta)
    np.save(os.path.join(out, "proj.npy"), np.concatenate(proj))
    with open(os.path.join(out, "files.json"), "w", encoding="utf-8") as f:
        json.dump(files, f)
    with open(os.path.join(out, "outfit_keys.json"), "w", encoding="utf-8") as f:
        json.dump(outfit_keys, f)


class CandidatePool:
    def __init__(self, *, stores, proj, emb, scalars, meta, files, outfit_keys, device):
        self.stores = list(stores)
        self.device = torch.device(device)
        self.n = len(files)
        self.files = files
        self.outfit_keys = outfit_keys
        for k in ("source", "split", "type", "broad", "store_row", "outfit_ord", "has_crop", "trend", "cluster",
                  "rep_demo", "rep_eval"):
            setattr(self, k, np.asarray(meta[k]))
        dev = self.device
        self.proj = torch.as_tensor(np.asarray(proj, np.float32), device=dev)
        e = torch.as_tensor(np.asarray(emb, np.float32), device=dev)
        self.emb = e / e.norm(dim=1, keepdim=True).clamp(min=1e-8)
        s = torch.as_tensor(np.asarray(scalars), dtype=torch.float64, device=dev)
        self.hue, self.formality, self.area = s[:, 0].contiguous(), s[:, 1].contiguous(), s[:, 2].contiguous()
        self.broad_t = torch.as_tensor(self.broad.astype(np.int64), device=dev)
        self.type_t = torch.as_tensor(self.type.astype(np.int64), device=dev)
        self.trend_t = torch.as_tensor(self.trend.astype(np.float32), device=dev)
        idx = np.arange(self.n)
        self._is_rep = {"demo": torch.as_tensor(self.rep_demo == idx, device=dev),
                        "eval": torch.as_tensor(self.rep_eval == idx, device=dev)}
        self._key_ord = {name: {k: i for i, k in enumerate(keys)} for name, keys in outfit_keys.items()}
        self._outfits = None

    @classmethod
    def load(cls, settings, device) -> "CandidatePool":
        d = pool_dir(settings)
        if not os.path.exists(os.path.join(d, "fingerprint.json")):
            raise FileNotFoundError(f"No Module 4 cache at {d}: run  python Module4/build_cache.py")
        if not is_fresh(settings):
            raise RuntimeError("Module 4 cache is stale (checkpoint, trend model, stores or t_dup changed): "
                               "run  python Module4/build_cache.py --force")
        emb, scal = load_store_arrays(settings)
        with open(os.path.join(d, "files.json"), encoding="utf-8") as f:
            files = json.load(f)
        with open(os.path.join(d, "outfit_keys.json"), encoding="utf-8") as f:
            keys = json.load(f)
        return cls(stores=settings.stores, proj=np.load(os.path.join(d, "proj.npy")), emb=emb, scalars=scal,
                   meta=dict(np.load(os.path.join(d, "meta.npz"))), files=files, outfit_keys=keys,
                   device=device)

    # ---- queries ----
    def empty_rows(self) -> torch.Tensor:
        return torch.zeros(0, dtype=torch.long, device=self.device)

    def candidate_rows(self, types, mode: str = "demo", exclude=None) -> torch.Tensor:
        """Representative rows (D8a) of the given specific types. mode "eval" = test rows only.
        exclude = (source name, outfit key): drop that outfit's own garments."""
        if not types:
            return self.empty_rows()
        tid = torch.tensor([TYPE_ID[t] for t in types], device=self.device)
        mask = torch.isin(self.type_t, tid) & self._is_rep[mode]
        if exclude is not None:
            own = self.outfit_rows(*exclude)
            mask[torch.as_tensor(own, device=self.device)] = False
        return torch.nonzero(mask).squeeze(1)

    def cosine_to(self, rows: torch.Tensor, v) -> torch.Tensor:
        v = torch.as_tensor(np.asarray(v, np.float32), device=self.device)
        return self.emb[rows] @ (v / v.norm().clamp(min=1e-8))

    def outfits(self) -> dict:
        """{(source_id, outfit_ord): np.ndarray of pool rows}."""
        if self._outfits is None:
            key = (self.source.astype(np.int64) << 32) | self.outfit_ord.astype(np.int64)
            order = np.argsort(key, kind="stable")
            cuts = np.flatnonzero(np.diff(key[order])) + 1
            self._outfits = {(int(self.source[g[0]]), int(self.outfit_ord[g[0]])): g
                             for g in np.split(order, cuts)}
        return self._outfits

    def source_id(self, name: str) -> int:
        return [s.name for s in self.stores].index(name)

    def outfit_rows(self, source: str, key: str) -> np.ndarray:
        return self.outfits()[(self.source_id(source), self._key_ord[source][key])]

    # ---- per-row metadata ----
    def source_name(self, row: int) -> str:
        return self.stores[int(self.source[row])].name

    def outfit_key(self, row: int) -> str:
        return self.outfit_keys[self.source_name(row)][int(self.outfit_ord[row])]

    def type_name(self, row: int) -> str:
        t = int(self.type[row])
        return TYPES[t] if t >= 0 else "unknown"

    def crop_path(self, row: int) -> str:
        """Repo-relative crop PNG path, rebuilt from the relative `file` field (never crop_path)."""
        spec = self.stores[int(self.source[row])]
        return "/".join([spec.output_root, self.outfit_key(row), self.files[row]])

    def describe(self, row: int) -> dict:
        return {"type": self.type_name(row), "hue": float(self.hue[row]),
                "formality": float(self.formality[row]), "area": float(self.area[row])}
