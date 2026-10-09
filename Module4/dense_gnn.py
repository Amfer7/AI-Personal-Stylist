"""Dense, broadcast re-implementation of CLIPOutfitGNN scoring for many same-size outfits.

Module 2 scores one outfit at a time through build_graph (a Python loop over edges). Module 4 needs
hundreds of thousands of what-if outfits per photo, so this file computes the same logits with
batched tensor ops:

- every outfit in a batch has the same number of nodes n, and every graph is complete (all i != j
  edges, as build_graph makes them), so edges live in a dense [B, n, n, 5] tensor indexed
  [batch, src, dst];
- the message is linear in the edge features, so sum_s w_sd * msg_edge(e_sd) is computed as
  (sum_s w_sd * e_sd) @ W.T + b * sum_s w_sd, which never builds a [B, n, n, 128] tensor;
- the input projection does not depend on the graph, so callers pass projected node states h0
  (CandidatePool.proj holds them for the whole pool).

Parity with model(build_graph(...)) is enforced by test_dense_gnn.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from m2_bridge import edge_weight_table


def edge_tensor(hue, formality, area, broad, wtable) -> torch.Tensor:
    """Dense 5-D edge features matching od.edge_features exactly.

    hue / formality / area: float64 [B, n] (hue NaN = unknown); broad: long [B, n] (BROAD_ID);
    wtable: float64 [V, V] from edge_weight_table(). Returns float32 [B, n(src), n(dst), 5] =
    [category weight, hue separation, formality difference, volume contrast, same category]."""
    d = (hue[:, :, None] - hue[:, None, :]).abs() % 360.0
    d = torch.where(d <= 180.0, d, 360.0 - d)
    hue_sep = torch.nan_to_num(d / 180.0, nan=0.0)
    form_diff = (formality[:, :, None] - formality[:, None, :]).abs() / 100.0
    vol = torch.tanh(torch.log((area[:, :, None] + 1.0) / (area[:, None, :] + 1.0)))
    w = wtable[broad[:, :, None], broad[:, None, :]]
    same = (broad[:, :, None] == broad[:, None, :]).to(torch.float64)
    return torch.stack([w, hue_sep, form_diff, vol, same], dim=-1).to(torch.float32)


def _layer(layer, h, E, offdiag):
    w = F.softplus(layer.gate(E)).squeeze(-1) * offdiag            # [B, src, dst], 0 on the diagonal
    wsum = w.sum(dim=1)                                             # [B, dst]
    agg = torch.einsum("bsd,bsk->bdk", w, h)
    agg = agg + torch.einsum("bsd,bsdc->bdc", w, E) @ layer.msg_edge.weight.T
    agg = agg + wsum[..., None] * layer.msg_edge.bias
    agg = agg / wsum.clamp(min=1e-6)[..., None]
    return F.relu(layer.norm(layer.linear(h + agg)))


class DenseGNN:
    def __init__(self, model, device, vram_gb: float = 6.0):
        self.model = model.eval()
        self.device = torch.device(device)
        self.wtable = edge_weight_table(self.device)
        self.cap_bytes = vram_gb * 2**30

    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """[N, 555] node inputs -> [N, 128] (the model's input_proj)."""
        return self.model.input_proj(x.to(self.device, torch.float32))

    @torch.no_grad()
    def logits(self, h0, hue, formality, area, broad) -> torch.Tensor:
        n = h0.shape[1]
        E = edge_tensor(hue, formality, area, broad, self.wtable)
        offdiag = ~torch.eye(n, dtype=torch.bool, device=h0.device)
        h = _layer(self.model.gnn1, h0, E, offdiag)
        h = _layer(self.model.gnn2, h, E, offdiag)
        return self.model.outfit_score(h.mean(dim=1)).squeeze(-1)

    def chunk_size(self, n: int) -> int:
        """Outfits per batch: half the VRAM left under the cap, at a conservative bytes/outfit."""
        per_outfit = n * 128 * 4 * 16 + n * n * 5 * 8 * 4
        if self.device.type == "cuda":
            budget = self.cap_bytes - torch.cuda.memory_allocated(self.device)
        else:
            budget = 2 * 2**30
        return int(max(1024, min(262144, 0.5 * budget // per_outfit)))

    def logits_chunked(self, make_batch, total: int, n: int, max_chunk=None) -> torch.Tensor:
        """make_batch(start, stop) -> (h0, hue, formality, area, broad) for outfits [start, stop).
        On CUDA OOM the chunk is halved and retried."""
        size = self.chunk_size(n) if max_chunk is None else max_chunk
        out, start = [], 0
        while start < total:
            stop = min(total, start + size)
            try:
                out.append(self.logits(*make_batch(start, stop)).float().cpu())
                start = stop
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size //= 2
        return torch.cat(out) if out else torch.zeros(0)
