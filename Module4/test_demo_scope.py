"""Query exclusions for the demo pool (demo_exclude_queries): which rows they remove."""
import numpy as np
import pandas as pd

from candidate_pool import excluded_by_query
from m4_settings import M4Settings, StoreSpec


def test_excludes_every_row_of_a_photo_with_any_banned_query(tmp_path):
    q = tmp_path / "queries.csv"
    pd.DataFrame({"key": ["a", "b", "b", "c"],
                  "query": ["saree draping", "wedding guest", "saree draping", "street style"]}).to_csv(q, index=False)
    stores = [StoreSpec("f144k", "s", "o", "x.mat"), StoreSpec("pin", "s", "o", "x.json", queries=str(q))]   # absolute: abs() keeps it
    s = M4Settings.load().with_(stores=stores, demo_exclude_queries=["saree draping"])
    meta = {"type": np.zeros(6, np.int8), "source": np.array([0, 0, 1, 1, 1, 1], np.uint8),
            "outfit_ord": np.array([0, 1, 0, 1, 1, 2], np.int32)}
    keys = {"f144k": ["a", "b"], "pin": ["a", "b", "c"]}       # same key names in another store: not excluded
    assert excluded_by_query(s, meta, keys).tolist() == [False, False, True, True, True, False]
    assert not excluded_by_query(s.with_(demo_exclude_queries=[]), meta, keys).any()
