"""pytest setup for Module 4: import path, the `real` marker, shared session fixtures.

Tests marked `real` need the Fashion144k data and (from Task 4 on) a built cache:
    python Module4/build_cache.py
Quick unit run: python -m pytest Module4 -m "not real"
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pytest_configure(config):
    config.addinivalue_line("markers", "real: needs the real data, a CUDA device and the built cache")


@pytest.fixture(scope="session")
def settings():
    from m4_settings import M4Settings
    return M4Settings.load()


@pytest.fixture(scope="session")
def device(settings):
    from m2_bridge import apply_vram_cap, pick_device
    d = pick_device()
    apply_vram_cap(settings.vram_gb, d)
    return d


@pytest.fixture(scope="session")
def gnn(settings, device):
    from dense_gnn import DenseGNN
    from m2_bridge import load_gnn
    return DenseGNN(load_gnn(settings.abs(settings.checkpoint), device), device, settings.vram_gb)


@pytest.fixture(scope="session")
def pool(settings, device):
    from candidate_pool import CandidatePool
    return CandidatePool.load(settings, device)


@pytest.fixture(scope="session")
def pool_features(settings, pool):
    """Feature-store rows by pool row, across every store: pool_features[rows] -> (len(rows), ROW_DIM)."""
    import numpy as np
    stores = [np.load(settings.abs(os.path.join(s.store_dir, "features.npy")), mmap_mode="r")
              for s in settings.stores]

    class _Features:
        def __getitem__(self, rows):
            rows = np.atleast_1d(np.asarray(rows))
            return np.stack([stores[pool.source[r]][pool.store_row[r]] for r in rows])

    return _Features()


@pytest.fixture(scope="session")
def recommender(settings, device, pool, gnn):
    from recommender import Recommender
    return Recommender.from_settings(settings, device, pool=pool, gnn=gnn)


@pytest.fixture(scope="session")
def image2_dir(settings):
    """M1 output for samples/image2.jpg, computed once into the gitignored cache."""
    out_root = settings.abs(os.path.join(settings.cache_dir, "runs"))
    d = os.path.join(out_root, "image2")
    if not os.path.exists(os.path.join(d, "metadata.json")):
        from run_recommend import segment_photo
        segment_photo(settings.abs("samples/image2.jpg"), out_root)
    return d
