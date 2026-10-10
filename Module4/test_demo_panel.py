"""Demo panel: what each widget shows, built from real Recommendations of the stub recommender."""
import os

from PIL import Image

from outfit_board import HIGHLIGHT, PAD, TILE

from demo_panel import (PER_PIECE_HEADERS, _change, also_markdown, choice_labels, choices, gallery_items,
                        outfit_tiles, per_piece_rows, swap_board, verdict_markdown)
from test_recommender_logic import make


def _swap():
    rec, st, _ = make()
    table = rec.score(st)
    return rec, table, rec.decide(table)


def test_swap_panel():
    rec, table, r = _swap()
    md = verdict_markdown(r)
    assert md.startswith("## Swap your shirt_blouse")
    assert "Outfit harmony" in md and "rule `topk_mean_gain`" in md
    items = gallery_items(r, rec.settings)
    assert len(items) == 3
    assert items[0][0] == rec.settings.abs(r.suggestions[0].crop_path)
    assert items[0][1].startswith("shirt_blouse → t_shirt")
    assert "skirt_0.png" in also_markdown(r)
    rows = per_piece_rows(r)
    assert len(rows) == 4 and all(len(x) == len(PER_PIECE_HEADERS) for x in rows)
    assert rows[0][0] == "shirt_blouse_0.png" and rows[0][6] == "✓"
    labels = choice_labels(r)
    assert len(labels) == len(choices(r)) == 4
    assert labels[0].startswith("1. shirt_blouse → t_shirt") and labels[3].startswith("Also: skirt_0.png")


def test_keep_panel_has_no_choices():
    rec, st, _ = make(trend_worn=100.0, tau={"max_gain": 99.0, "topk_mean_gain": 99.0, "slot_percentile": 99.0})
    r = rec.decide(rec.score(st))
    assert r.verdict == "keep"
    assert verdict_markdown(r).startswith("## Already strong, no swap recommended")
    assert choice_labels(r) == [] and gallery_items(r, rec.settings) == [] and also_markdown(r) == ""


def test_swap_board_replaces_the_right_piece(tmp_path):
    rec, table, r = _swap()
    out, text = swap_board(r, table.state, rec.pool, rec.settings, 0, str(tmp_path / "b.png"))
    assert os.path.exists(out)
    assert Image.open(out).size[0] == 12 + 4 * (200 + 12)        # 4 pieces, one replaced
    assert text.startswith("shirt_blouse → t_shirt") and "Harmony" in text
    tiles = outfit_tiles(table.state, rec.pool, rec.settings)
    assert [t[1] for t in tiles] == ["shirt_blouse", "skirt", "leggings", "shoe"]
    im = Image.open(out).convert("RGB")
    assert im.getpixel((PAD + 1, PAD + 1)) == HIGHLIGHT                       # swapped tile (index 0)
    assert im.getpixel((PAD + TILE + PAD + 1, PAD + 1)) != HIGHLIGHT          # an unswapped tile


def test_change_marks_same_type():
    assert _change("shirt", "t_shirt") == "shirt → t_shirt"
    assert _change("shirt", "shirt") == "shirt (same type)"
