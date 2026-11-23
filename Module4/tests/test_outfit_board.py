"""Outfit board: layout, highlight and the missing-crop placeholder."""
from PIL import Image

from outfit_board import BORDER, CAPTION_H, HIGHLIGHT, PAD, PLACEHOLDER, TILE, outfit_board


def _crop(path, size, colour):
    Image.new("RGBA", size, colour + (255,)).save(path)
    return str(path)


def test_board_layout_highlight_and_placeholder(tmp_path):
    red = _crop(tmp_path / "a.png", (100, 300), (200, 30, 30))      # tall crop: scaled to fit
    blue = _crop(tmp_path / "b.png", (120, 80), (30, 30, 200))
    out = outfit_board([(red, "pants", False), (blue, "NEW bag", True), (None, "shoe", False)],
                       str(tmp_path / "board.png"))
    im = Image.open(out).convert("RGB")
    assert im.size == (PAD + 3 * (TILE + PAD), PAD + TILE + CAPTION_H + PAD)
    x0, y = PAD, PAD
    assert im.getpixel((x0 + TILE // 2, y + TILE // 2)) == (200, 30, 30)        # crop pasted, centred
    x1 = PAD + (TILE + PAD)
    assert im.getpixel((x1 + 1, y + 1)) == HIGHLIGHT                           # highlighted tile
    assert im.getpixel((x1 + BORDER + 2, y + BORDER + 2)) == (255, 255, 255)   # inside the border: white
    x2 = PAD + 2 * (TILE + PAD)
    assert im.getpixel((x2 + 5, y + 5)) == PLACEHOLDER                         # missing crop


def test_title_adds_a_band(tmp_path):
    out = outfit_board([(None, "hat", False)], str(tmp_path / "t.png"), title="Swap your hat")
    assert Image.open(out).size[1] == 36 + PAD + TILE + CAPTION_H + PAD


def test_corrupt_image_draws_placeholder(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    out = outfit_board([(str(bad), "x", False)], str(tmp_path / "c.png"))
    assert Image.open(out).convert("RGB").getpixel((PAD + 5, PAD + 5)) == PLACEHOLDER


def test_upscale_enlarges_small_crops_to_the_tile(tmp_path):
    small = _crop(tmp_path / "s.png", (40, 20), (30, 160, 30))
    plain = Image.open(outfit_board([(small, "", False)], str(tmp_path / "p.png"))).convert("RGB")
    big = Image.open(outfit_board([(small, "", False)], str(tmp_path / "u.png"), upscale=True)).convert("RGB")
    edge = PAD + BORDER + 2                         # near the tile's left edge, mid-height
    mid = PAD + TILE // 2
    assert plain.getpixel((edge, mid)) == (255, 255, 255)      # default: a 40-px crop stays small
    assert big.getpixel((edge, mid)) == (30, 160, 30)          # upscale: it fills the tile's width
