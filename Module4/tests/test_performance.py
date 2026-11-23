"""D5 gate: samples/image2.jpg, full demo pool, <= 5 s warm, peak VRAM <= 6 GB.
Run on its own for a clean VRAM reading:  python -m pytest Module4/tests/test_performance.py -s"""
import statistics
import time

import pytest
import torch

pytestmark = pytest.mark.real


def test_image2_full_demo_pool_within_gate(recommender, image2_dir, settings):
    recommender.recommend(image2_dir)                      # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        t = time.perf_counter()
        recommender.recommend(image2_dir)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    med = statistics.median(times)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"\n[perf] warm recommend(): median {med:.2f}s (runs {', '.join(f'{x:.2f}' for x in times)}), "
          f"peak VRAM {peak:.2f} GiB")
    assert med <= 5.0
    assert peak <= settings.vram_gb
