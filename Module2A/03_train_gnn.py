"""
03_train_gnn.py
================
Stage 6-7 of the plan doc: the CLIP-native OutfitGNN (replaces demo4.py's
nn.Embedding(vocab_size,...) input with a Linear projection of real 512-D CLIP
embeddings + 32-D colour histograms), trained with fashionability-weighted BPR
loss against real Fashion144K outfit graphs, evaluated by AUC on the official
test split.

Usage (Colab):
    !python 03_train_gnn.py \
        --output_root /content/drive/MyDrive/Fashion144k/outputs \
        --split_mat   /content/drive/MyDrive/Fashion144k/split.mat \
        --relvotes_mat /content/drive/MyDrive/Fashion144k/relvotes.mat \
        --checkpoint_dir /content/drive/MyDrive/Fashion144k/gnn_checkpoints \
        --epochs 5 --lr 1e-4
"""

import os
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.util import spec_from_file_location, module_from_spec

_ds_spec = spec_from_file_location(
    "outfit_dataset", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_outfit_dataset.py")
)
outfit_dataset = module_from_spec(_ds_spec)
_ds_spec.loader.exec_module(outfit_dataset)

load_split = outfit_dataset.load_split
load_relvotes = outfit_dataset.load_relvotes
build_category_pool = outfit_dataset.build_category_pool
OutfitDataset = outfit_dataset.OutfitDataset
NODE_ATTR_DIM = outfit_dataset.NODE_ATTR_DIM
EDGE_ATTR_DIM = outfit_dataset.EDGE_ATTR_DIM
build_node_from_meta = outfit_dataset.build_node_from_meta


# ---------------- model (CLIP-native version of demo4.py's OutfitGNN) ----------------

class EdgeAwareGNNLayer(nn.Module):
    """Weighted message passing where both the aggregation weight and the message content
    are functions of the edge features (category weight, hue separation, formality
    difference, volume contrast, same-category). Generalises the original scalar-weighted
    layer: the fixed category weight is now one component of a learned edge gate."""

    def __init__(self, dim, edge_dim):
        super().__init__()
        self.msg_edge = nn.Linear(edge_dim, dim)   # inject edge cues into the message
        self.gate = nn.Linear(edge_dim, 1)         # learned per-edge weight
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr, num_nodes):
        if edge_index.shape[1] == 0:
            return F.relu(self.norm(self.linear(x)))

        src, dst = edge_index
        w = F.softplus(self.gate(edge_attr)).squeeze(-1)        # [E], positive
        msg = x[src] + self.msg_edge(edge_attr)                 # [E, dim]
        weighted_msgs = msg * w.unsqueeze(1)

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, weighted_msgs)

        weight_sum = torch.zeros(num_nodes, device=x.device)
        weight_sum.index_add_(0, dst, w)
        weight_sum = weight_sum.clamp(min=1e-6).unsqueeze(1)

        agg = agg / weight_sum
        return F.relu(self.norm(self.linear(x + agg)))


class CLIPOutfitGNN(nn.Module):
    """Node input = [512-D CLIP embedding | 32-D colour histogram | attribute features]
    -> Linear projection -> EdgeAwareGNNLayer x2 -> per-item + outfit scores.
    No nn.Embedding / vocab lookup -- this is the fix for the train/inference
    representation mismatch described in the plan doc. Edges carry the fashion-theory
    cues from Module 1."""

    def __init__(self, clip_dim=512, colour_dim=32, attr_dim=NODE_ATTR_DIM,
                 edge_dim=EDGE_ATTR_DIM, hidden_dim=128):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(clip_dim + colour_dim + attr_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gnn1 = EdgeAwareGNNLayer(hidden_dim, edge_dim)
        self.gnn2 = EdgeAwareGNNLayer(hidden_dim, edge_dim)

        self.item_score = nn.Linear(hidden_dim, 1)
        self.outfit_score = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        N = x.shape[0]
        h = self.input_proj(x)
        h = self.gnn1(h, edge_index, edge_attr, N)
        h = self.gnn2(h, edge_index, edge_attr, N)

        per_item = self.item_score(h).squeeze(-1)

        if batch is None:
            # single-graph path: mean-pool all nodes -> scalar score
            outfit_repr = h.mean(dim=0, keepdim=True)
            return self.outfit_score(outfit_repr).squeeze(), per_item

        # batched path: segment mean-pool by outfit id -> one score per outfit
        B = int(batch.max().item()) + 1
        sums = torch.zeros(B, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, batch, h)
        counts = torch.zeros(B, device=h.device, dtype=h.dtype).index_add_(
            0, batch, torch.ones_like(batch, dtype=h.dtype)).clamp(min=1.0).unsqueeze(1)
        outfit_repr = sums / counts
        scores = self.outfit_score(outfit_repr).squeeze(-1)  # [B]
        return scores, per_item


# ---------------- BPR negative sampling ----------------

def make_negative_sample(dataset, nodes, category_pool, current_idx):
    """Copy the positive outfit's node list, swap ONE node for a same-category garment
    drawn from a DIFFERENT outfit. The substitute node (embedding + colour + attributes)
    comes from the feature store when available, else disk."""
    N = len(nodes)
    swap_i = random.randrange(N)
    cat = nodes[swap_i]["category"]

    # Rejection sampling instead of materialising a filtered copy of the whole pool.
    # The old `[c for c in pool if c[0] != current_idx]` was O(pool size) *per negative*;
    # with full-data pools (~100k) that starved the GPU (3% util, ~27 min/epoch). Picking a
    # random candidate and retrying only on the rare same-outfit collision is O(1) amortised
    # and draws from the identical distribution (uniform over same-category garments not in
    # the current outfit).
    pool = category_pool.get(cat)
    if not pool:
        return None  # no garments of this category anywhere
    sub = None
    for _ in range(8):
        cand = random.choice(pool)
        if cand[0] != current_idx:
            sub = cand
            break
    if sub is None:
        return None  # pool is (almost) entirely this outfit -- effectively no substitute

    sub_idx, sub_filename = sub
    try:
        sub_node = dataset.build_node(sub_idx, sub_filename, broad_category=cat)
    except (KeyError, StopIteration, FileNotFoundError):
        return None

    neg_nodes = list(nodes)
    neg_nodes[swap_i] = sub_node
    return neg_nodes


# ---------------- training loop ----------------

def resolve_attr_mask(args, use_attributes):
    """Returns the attr-subset column index list when --attr_subset is set, else None.
    Errors if combined with --no_attributes (there are no node attrs to subset)."""
    if not getattr(args, "attr_subset", False):
        return None
    if not use_attributes:
        raise SystemExit("[ERROR] --attr_subset cannot be combined with --no_attributes "
                         "(the baseline has no attribute node features to subset).")
    return outfit_dataset.attr_subset_indices()


def set_seed(seed):
    """Seed all RNGs so runs (and the with/without-attributes comparison) are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device = {device}")
    set_seed(args.seed)
    print(f"[INFO] seed = {args.seed}")

    split = load_split(args.split_mat)
    relvotes = load_relvotes(args.relvotes_mat)

    # Pilot runs: cap the number of *train* outfits (deterministic first-N slice, so it is
    # reproducible across seeds). Val/test are left full so the reported AUC is still honest.
    if args.limit_train is not None:
        split["train"] = split["train"][:args.limit_train]
        print(f"[INFO] PILOT: capping train set to first {len(split['train'])} outfit ids")

    use_attributes = not args.no_attributes
    print(f"[INFO] use_attributes = {use_attributes} "
          f"({'full attr+edge features' if use_attributes else 'ABLATION: CLIP+colour only, scalar edges'})")

    attr_mask = resolve_attr_mask(args, use_attributes)
    if attr_mask is not None:
        print(f"[INFO] ATTR SUBSET ablation: keeping {len(attr_mask)}/{NODE_ATTR_DIM} attr dims "
              f"(silhouette + scalars; dropping pattern/fabric/fit one-hots)")

    train_ds = OutfitDataset(split["train"], args.output_root, relvotes, feature_store=args.feature_store)
    val_ds = OutfitDataset(split["val"], args.output_root, relvotes, feature_store=args.feature_store)
    print(f"[INFO] {len(train_ds)} usable train outfits, {len(val_ds)} usable val outfits")
    print(f"[INFO] feature_store = {args.feature_store or 'None (reading from disk)'}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if train_ds.store is not None:
        category_pool = train_ds.build_pool(train_ds.valid_indices)
    else:
        category_pool = build_category_pool(train_ds.valid_indices, args.output_root,
                                            cache_path=os.path.join(args.checkpoint_dir, "category_pool.pkl"))
    print(f"[INFO] category pool sizes: {[(k, len(v)) for k, v in category_pool.items()]}")

    if not use_attributes:
        attr_dim = 0
    elif attr_mask is not None:
        attr_dim = len(attr_mask)
    else:
        attr_dim = NODE_ATTR_DIM
    edge_dim = EDGE_ATTR_DIM if use_attributes else 1
    model = CLIPOutfitGNN(attr_dim=attr_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    order = list(range(len(train_ds)))
    step = 0
    best_val, best_state, best_epoch = -1.0, None, -1

    for epoch in range(args.epochs):
        random.shuffle(order)
        model.train()
        running_loss, nb = 0.0, 0

        # Tier 2: process B outfits per step as one batched (disconnected) graph.
        # Tier 3: K negatives per positive (num_negatives).
        for start in range(0, len(order), args.batch_size):
            batch_ids = order[start:start + args.batch_size]
            pos_graphs, weights = [], []
            neg_graphs, neg_owner = [], []   # neg_owner[j] -> index of its positive
            for ds_i in batch_ids:
                idx = train_ds.valid_indices[ds_i]
                nodes = train_ds.load_outfit_nodes(idx)
                fs = float(relvotes[idx]) / 10.0  # 1-10 -> 0.1-1.0
                negs = []
                for _ in range(args.num_negatives):
                    ng = make_negative_sample(train_ds, nodes, category_pool, idx)
                    if ng is not None:
                        negs.append(ng)
                if not negs:
                    continue
                pi = len(pos_graphs)
                pos_graphs.append(train_ds.build_graph(nodes, fs, use_attributes=use_attributes,
                                                       attr_mask=attr_mask))
                weights.append(fs)
                for ng in negs:
                    neg_graphs.append(train_ds.build_graph(ng, fs, use_attributes=use_attributes,
                                                           attr_mask=attr_mask))
                    neg_owner.append(pi)
            if not pos_graphs:
                continue

            posb = train_ds.collate(pos_graphs)
            negb = train_ds.collate(neg_graphs)
            pos_scores, _ = model(posb["x"].to(device), posb["edge_index"].to(device),
                                  posb["edge_attr"].to(device), posb["batch"].to(device))
            neg_scores, _ = model(negb["x"].to(device), negb["edge_index"].to(device),
                                  negb["edge_attr"].to(device), negb["batch"].to(device))
            owner = torch.tensor(neg_owner, dtype=torch.long, device=device)
            w = torch.tensor(weights, dtype=torch.float32, device=device)[owner]
            loss = (-w * F.logsigmoid(pos_scores[owner] - neg_scores)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            nb += 1
            step += 1
            if step % 50 == 0:
                print(f"[E{epoch} S{step}] BPR loss = {running_loss / nb:.4f}")
                running_loss, nb = 0.0, 0

        val_auc = evaluate(model, val_ds, category_pool, device,
                            max_outfits=args.val_eval_max, use_attributes=use_attributes,
                            attr_mask=attr_mask)
        print(f"[EPOCH {epoch}] val AUC = {val_auc:.4f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"clip_outfit_gnn_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[SAVED] {ckpt_path}")

        # Track the best-val model so the final test uses it (not the last epoch).
        if val_auc == val_auc and val_auc > best_val:  # val_auc==val_auc filters NaN
            best_val, best_epoch = val_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "clip_outfit_gnn_best.pt"))

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] restored best-val model (epoch {best_epoch}, val AUC {best_val:.4f})")

    return model


@torch.no_grad()
def evaluate(model, dataset, category_pool, device, max_outfits=None, use_attributes=True,
             attr_mask=None):
    """AUC over pos-vs-neg score pairs: label 1 for the positive outfit's score,
    label 0 for the corrupted negative's score, across sampled outfits."""
    model.eval()
    indices = dataset.valid_indices
    if max_outfits is not None and len(indices) > max_outfits:
        indices = random.sample(indices, max_outfits)

    scores, labels = [], []
    buf_pos, buf_neg = [], []
    eval_bs = 256

    def flush():
        if not buf_pos:
            return
        pb = dataset.collate(buf_pos)
        nb = dataset.collate(buf_neg)
        ps, _ = model(pb["x"].to(device), pb["edge_index"].to(device),
                      pb["edge_attr"].to(device), pb["batch"].to(device))
        ns, _ = model(nb["x"].to(device), nb["edge_index"].to(device),
                      nb["edge_attr"].to(device), nb["batch"].to(device))
        ps = torch.sigmoid(ps).cpu().numpy().reshape(-1)
        ns = torch.sigmoid(ns).cpu().numpy().reshape(-1)
        for a, b in zip(ps, ns):
            scores.extend([float(a), float(b)])
            labels.extend([1, 0])
        buf_pos.clear()
        buf_neg.clear()

    for idx in indices:
        nodes = dataset.load_outfit_nodes(idx)
        neg_nodes = make_negative_sample(dataset, nodes, category_pool, idx)
        if neg_nodes is None:
            continue
        buf_pos.append(dataset.build_graph(nodes, 0.0, use_attributes=use_attributes,
                                           attr_mask=attr_mask))
        buf_neg.append(dataset.build_graph(neg_nodes, 0.0, use_attributes=use_attributes,
                                           attr_mask=attr_mask))
        if len(buf_pos) >= eval_bs:
            flush()
    flush()

    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--split_mat", required=True)
    parser.add_argument("--relvotes_mat", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Outfits per batched-graph training step (Tier 2)")
    parser.add_argument("--num_negatives", type=int, default=1,
                        help="Negatives sampled per positive outfit (Tier 3)")
    parser.add_argument("--val_eval_max", type=int, default=1000,
                        help="Max val outfits sampled for the per-epoch AUC (Tier 3)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible runs")
    parser.add_argument("--limit_train", type=int, default=None,
                        help="Pilot runs: use only the first N train outfits (val/test stay full)")
    parser.add_argument("--feature_store", default=None,
                        help="Path to a prebuilt feature store dir (build_feature_store.py). "
                             "If set, training reads features from the memmap instead of PNG/npy.")
    parser.add_argument("--no_attributes", action="store_true",
                        help="Ablation: train on CLIP+colour only with scalar category edges "
                             "(disables the Module 1 attribute node features and fashion-theory "
                             "edge features).")
    parser.add_argument("--attr_subset", action="store_true",
                        help="Ablation: keep only the reliable attr dims (silhouette one-hot + "
                             "the 7 scalars), dropping the noisy CLIP zero-shot pattern/fabric/fit "
                             "one-hots. Edges stay 5-D. Cannot combine with --no_attributes.")
    args = parser.parse_args()

    model = train(args)

    split = load_split(args.split_mat)
    relvotes = load_relvotes(args.relvotes_mat)
    test_ds = OutfitDataset(split["test"], args.output_root, relvotes, feature_store=args.feature_store)
    if test_ds.store is not None:
        pool = test_ds.build_pool(test_ds.valid_indices)
    else:
        pool = build_category_pool(
            test_ds.valid_indices, args.output_root,
            cache_path=os.path.join(args.checkpoint_dir, "category_pool_test.pkl"),
        )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_attributes = not args.no_attributes
    attr_mask = resolve_attr_mask(args, use_attributes)
    set_seed(args.seed)  # identical test negative-sampling across ablation arms
    test_auc = evaluate(model, test_ds, pool, device,
                        use_attributes=use_attributes, attr_mask=attr_mask)
    if args.no_attributes:
        tag = "no_attributes"
    elif args.attr_subset:
        tag = "attr_subset"
    else:
        tag = "with_attributes"
    print(f"[FINAL] Test AUC on official split.mat testids ({tag}): {test_auc:.4f}")


if __name__ == "__main__":
    main()
