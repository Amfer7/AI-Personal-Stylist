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

POOL_FORMAT = 3                     # 3: outfit-level repost groups (D8a)
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
        entry = {"name": s.name, "rows": int(feat.shape[0]), "split": _sha16(settings.abs(s.split)),
                 "output_entries": crops}
        if s.product_photos:     # embeddings are rewritten in place by hm_catalogue.py --align
            entry["features"] = _sha16(settings.abs(os.path.join(s.store_dir, "features.npy")))
        stores.append(entry)
    return {"format": POOL_FORMAT, "checkpoint": _sha16(settings.abs(settings.checkpoint)),
            "trend_model": _sha16(settings.abs(settings.trend_model)), "t_dup": settings.t_dup,
            "dup_max_colour_l1": settings.dup_max_colour_l1, "dup_max_area_ratio": settings.dup_max_area_ratio,
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


def load_store_colour(settings) -> np.ndarray:
    """Colour features [N, 32] (three histograms, each summing to 1), pool row order."""
    parts = []
    for s in settings.stores:
        feat = np.load(os.path.join(settings.abs(s.store_dir), "features.npy"), mmap_mode="r")
        parts.append(np.ascontiguousarray(feat[:, 512:544]).astype(np.float32))
    return np.concatenate(parts)


def is_duplicate(cos, colour_l1, area_a, area_b, t_dup: float, max_l1: float, max_ratio: float):
    """The near-duplicate rule (D8a/b): CLIP cosine >= t_dup AND matching colour histograms AND crop
    areas within max_ratio. CLIP alone is not enough: on tiny, mostly empty crops (e.g. a left and a
    right shoe far apart) it collapses, and thousands of distinct items score >= 0.99."""
    big, small = torch.maximum(area_a, area_b), torch.minimum(area_a, area_b).clamp(min=1.0)
    return (cos >= t_dup) & (colour_l1 <= max_l1) & (big / small <= max_ratio)


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


def repost_groups(emb: torch.Tensor, colour: torch.Tensor, type_ids: np.ndarray, outfit: np.ndarray,
                  area: np.ndarray, t_dup: float, max_l1: float, max_ratio: float, chunk: int = 2048) -> np.ndarray:
    """Repost groups (D8a): outfits that are the same photo posted more than once. Two outfits are reposts
    when both have the same number of typed garments (at least 2) and every garment has a duplicate
    (is_duplicate) of the same type in the other outfit. Matching single garments is not enough: on
    tiny, mostly empty crops CLIP, colour and size all collapse, so distinct items match.
    Groups do not chain: outfits are visited in order; an unassigned outfit claims the unassigned
    outfits that are reposts of it directly. outfit: dense outfit id per row (-1 = ignore).
    Returns the group id per row: the id of the group's first outfit (-1 where outfit is -1)."""
    from scipy.sparse import csr_matrix

    typed = (type_ids >= 0) & (outfit >= 0)
    n_out = int(outfit.max()) + 1
    n_g = np.bincount(outfit[typed], minlength=n_out)
    dev = emb.device
    area_t = torch.as_tensor(np.asarray(area, np.float64), device=dev)
    keys = []
    for t in np.unique(type_ids[typed]):
        rows = np.nonzero(typed & (type_ids == t))[0]
        r_t = torch.as_tensor(rows, device=dev)
        E, C, A = emb[r_t], colour[r_t], area_t[r_t]
        for s in range(0, len(rows), chunk):
            ii, jj = torch.nonzero(E[s:s + chunk] @ E.T >= t_dup, as_tuple=True)
            ii = ii + s
            keep = jj > ii
            ii, jj = ii[keep], jj[keep]
            ok = is_duplicate(torch.ones_like(ii, dtype=E.dtype), (C[ii] - C[jj]).abs().sum(1), A[ii], A[jj],
                              t_dup, max_l1, max_ratio)
            oa, ob = outfit[rows[ii[ok].cpu().numpy()]], outfit[rows[jj[ok].cpu().numpy()]]
            diff = oa != ob
            lo, hi = np.minimum(oa, ob)[diff].astype(np.int64), np.maximum(oa, ob)[diff].astype(np.int64)
            keys.append(np.unique(lo * n_out + hi))          # one garment per type per outfit: count once
    keys = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    pair, count = np.unique(keys, return_counts=True)
    a, b = pair // n_out, pair % n_out
    full = (n_g[a] >= 2) & (n_g[a] == n_g[b]) & (count == n_g[a])
    a, b = a[full], b[full]
    adj = csr_matrix((np.ones(2 * len(a), np.int8), (np.r_[a, b], np.r_[b, a])), shape=(n_out, n_out))
    group = np.full(n_out, -1, np.int64)
    for o in np.unique(np.r_[a, b]):
        if group[o] >= 0:
            continue
        group[o] = o
        nb = adj.indices[adj.indptr[o]:adj.indptr[o + 1]]
        group[nb[group[nb] < 0]] = o
    alone = group < 0
    group[alone] = np.nonzero(alone)[0]
    out = np.full(len(type_ids), -1, np.int32)
    out[outfit >= 0] = group[outfit[outfit >= 0]]
    return out


def representatives(repost: np.ndarray, type_ids: np.ndarray, area: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """One representative per (repost group, garment type) among eligible rows: the largest crop (ties:
    lowest row). Returns rep_of int32 [N], -1 = not eligible."""
    rep = np.full(len(type_ids), -1, np.int32)
    rows = np.nonzero(eligible & (type_ids >= 0) & (repost >= 0))[0]
    if not len(rows):
        return rep
    key = repost[rows].astype(np.int64) * 256 + type_ids[rows].astype(np.int64)
    order = np.lexsort((rows, -np.asarray(area, np.float64)[rows], key))
    k = key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    grp = np.cumsum(first) - 1
    rep[rows[order]] = rows[order][first][grp]
    return rep


def excluded_by_query(settings, meta: dict, outfit_keys: dict) -> np.ndarray:
    """bool [N]: rows whose outfit photo was found by any query in settings.demo_exclude_queries
    (stores without a `queries` file have none excluded)."""
    out = np.zeros(len(meta["type"]), bool)
    banned = set(settings.demo_exclude_queries)
    if not banned:
        return out
    import pandas as pd
    for sid, spec in enumerate(settings.stores):
        if not spec.queries:
            continue
        q = pd.read_csv(settings.abs(spec.queries), dtype=str)
        bad_keys = set(q.loc[q["query"].isin(banned), "key"])
        bad_ord = np.array([k in bad_keys for k in outfit_keys[spec.name]], bool)
        rows = meta["source"] == sid
        out[rows] = bad_ord[meta["outfit_ord"][rows]]
    return out


def gate_demo_pool(settings, meta: dict, scalars: np.ndarray, outfit_keys: dict | None = None) -> dict:
    """Recompute rep_demo with the crop quality gate (crop_quality.py), the query exclusions
    (demo_exclude_queries) and demo_sources. Applied at load time, so a change needs no rebuild;
    meta.npz keeps the ungated rep_demo. Product-photo stores skip the gate's CLIP checks."""
    from crop_quality import crop_ok, load_metrics
    if outfit_keys is None:
        with open(os.path.join(pool_dir(settings), "outfit_keys.json"), encoding="utf-8") as f:
            outfit_keys = json.load(f)
    sid = lambda pick: [i for i, s in enumerate(settings.stores) if pick(s)]
    product = np.isin(meta["source"], sid(lambda s: s.product_photos))
    ok = crop_ok(load_metrics(settings, pool_dir(settings)), meta["type"], settings.crop_gate, shape_only=product)
    shown = np.isin(meta["source"], sid(lambda s: s.name in settings.demo_sources))
    eligible = (meta["type"] >= 0) & meta["has_crop"] & ok & shown & ~excluded_by_query(settings, meta, outfit_keys)
    return {**meta, "rep_demo": representatives(meta["repost"], meta["type"], scalars[:, 2], eligible)}


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
    col_t = torch.as_tensor(load_store_colour(settings), device=gnn.device)
    dup = (settings.t_dup, settings.dup_max_colour_l1, settings.dup_max_area_ratio)
    outfit = np.unique((meta["source"].astype(np.int64) << 32) | meta["outfit_ord"], return_inverse=True)[1]
    meta["repost"] = repost_groups(emb_t, col_t, meta["type"], outfit.astype(np.int64), scal[:, 2], *dup)
    meta["rep_demo"] = representatives(meta["repost"], meta["type"], scal[:, 2], known & meta["has_crop"])
    meta["rep_eval"] = representatives(meta["repost"], meta["type"], scal[:, 2], known & test)
    del emb_t, col_t
    np.savez(os.path.join(out, "meta.npz"), **meta)
    np.save(os.path.join(out, "proj.npy"), np.concatenate(proj))
    with open(os.path.join(out, "files.json"), "w", encoding="utf-8") as f:
        json.dump(files, f)
    with open(os.path.join(out, "outfit_keys.json"), "w", encoding="utf-8") as f:
        json.dump(outfit_keys, f)


class CandidatePool:
    def __init__(self, *, stores, proj, emb, scalars, meta, files, outfit_keys, device, colour=None, dup_rule=None):
        self.stores = list(stores)
        self.device = torch.device(device)
        self.n = len(files)
        self.files = files
        self.outfit_keys = outfit_keys
        for k in ("source", "split", "type", "broad", "store_row", "outfit_ord", "has_crop", "trend", "cluster",
                  "rep_demo", "rep_eval", "repost"):
            setattr(self, k, np.asarray(meta[k]))
        dev = self.device
        self.proj = torch.as_tensor(np.asarray(proj, np.float32), device=dev)
        e = torch.as_tensor(np.asarray(emb, np.float32), device=dev)
        self.emb = e / e.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.colour = torch.as_tensor(np.zeros((self.n, 32), np.float32) if colour is None
                                      else np.asarray(colour, np.float32), device=dev)
        self.dup_rule = dup_rule or (0.99, 0.25, 1.25)       # (t_dup, max colour L1, max area ratio)
        s = torch.as_tensor(np.asarray(scalars), dtype=torch.float64, device=dev)
        self.hue, self.formality, self.area = s[:, 0].contiguous(), s[:, 1].contiguous(), s[:, 2].contiguous()
        self.broad_t = torch.as_tensor(self.broad.astype(np.int64), device=dev)
        self.type_t = torch.as_tensor(self.type.astype(np.int64), device=dev)
        self.trend_t = torch.as_tensor(self.trend.astype(np.float32), device=dev)
        self.repost_t = torch.as_tensor(self.repost.astype(np.int64), device=dev)
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
            raise RuntimeError("Module 4 cache is stale (checkpoint, trend model, stores or duplicate rule changed): "
                               "run  python Module4/build_cache.py --force")
        emb, scal = load_store_arrays(settings)
        with open(os.path.join(d, "files.json"), encoding="utf-8") as f:
            files = json.load(f)
        with open(os.path.join(d, "outfit_keys.json"), encoding="utf-8") as f:
            keys = json.load(f)
        meta = gate_demo_pool(settings, dict(np.load(os.path.join(d, "meta.npz"))), scal, keys)
        return cls(stores=settings.stores, proj=np.load(os.path.join(d, "proj.npy")), emb=emb, scalars=scal,
                   meta=meta, files=files, outfit_keys=keys,
                   device=device, colour=load_store_colour(settings),
                   dup_rule=(settings.t_dup, settings.dup_max_colour_l1, settings.dup_max_area_ratio))

    # ---- queries ----
    def empty_rows(self) -> torch.Tensor:
        return torch.zeros(0, dtype=torch.long, device=self.device)

    def candidate_rows(self, types, mode: str = "demo", exclude=None) -> torch.Tensor:
        """Representative rows (D8a) of the given specific types. mode "eval" = test rows only.
        exclude = (source name, outfit key): drop that outfit's garments and those of its reposts."""
        if not types:
            return self.empty_rows()
        tid = torch.tensor([TYPE_ID[t] for t in types], device=self.device)
        mask = torch.isin(self.type_t, tid) & self._is_rep[mode]
        if exclude is not None:
            own = self.outfit_rows(*exclude)
            mask[torch.as_tensor(own, device=self.device)] = False
            mask &= self.repost_t != int(self.repost[own[0]])
        return torch.nonzero(mask).squeeze(1)

    def cosine_to(self, rows: torch.Tensor, v) -> torch.Tensor:
        v = torch.as_tensor(np.asarray(v, np.float32), device=self.device)
        return self.emb[rows] @ (v / v.norm().clamp(min=1e-8))

    def dup_mask(self, rows: torch.Tensor, emb, colour, area: float) -> torch.Tensor:
        """Which of `rows` are near-duplicates (is_duplicate) of a garment with this embedding, colour
        features and area (area in the 98,304-px frame, like the pool)."""
        if not len(rows):
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        c = torch.as_tensor(np.zeros(32, np.float32) if colour is None else np.asarray(
            colour.cpu() if torch.is_tensor(colour) else colour, np.float32), device=self.device)
        a = torch.full((len(rows),), float(area), dtype=torch.float64, device=self.device)
        e = emb.cpu().numpy() if torch.is_tensor(emb) else emb
        return is_duplicate(self.cosine_to(rows, e), (self.colour[rows] - c).abs().sum(1), self.area[rows], a,
                            *self.dup_rule)

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
