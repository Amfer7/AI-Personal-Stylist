"""GNN scoring of an outfit and of every what-if swap of one piece, through DenseGNN.

The fixed pieces are broadcast across the batch and only slot i changes, so building B what-if
outfits is one gather from the pool's precomputed tensors (no Python graph loop).
"""
from __future__ import annotations

import numpy as np
import torch

from m2_bridge import BROAD_ID


class SwapScorer:
    def __init__(self, pool, gnn):
        self.pool, self.gnn = pool, gnn

    def _fixed(self, state):
        dev = self.gnn.device
        ps = state.pieces
        h = torch.stack([p.proj.to(dev) for p in ps])
        f64 = lambda xs: torch.tensor(xs, dtype=torch.float64, device=dev)
        return (h, f64([p.hue for p in ps]), f64([p.formality for p in ps]), f64([p.area for p in ps]),
                torch.tensor([BROAD_ID[p.broad] for p in ps], device=dev))

    def outfit_logit(self, state) -> float:
        h, hue, form, area, broad = self._fixed(state)
        return float(self.gnn.logits(h[None], hue[None], form[None], area[None], broad[None])[0])

    def swap_logits(self, state, i: int, rows: torch.Tensor) -> np.ndarray:
        h, hue, form, area, broad = self._fixed(state)
        n, pool = h.shape[0], self.pool

        def make(a, b):
            r = rows[a:b]
            B = r.shape[0]
            H = h.expand(B, n, -1).clone()
            H[:, i] = pool.proj[r]
            out = [H]
            for fixed, col in ((hue, pool.hue), (form, pool.formality), (area, pool.area), (broad, pool.broad_t)):
                x = fixed.expand(B, n).clone()
                x[:, i] = col[r]
                out.append(x)
            return tuple(out)

        return self.gnn.logits_chunked(make, int(rows.shape[0]), n).numpy().astype(np.float32)
