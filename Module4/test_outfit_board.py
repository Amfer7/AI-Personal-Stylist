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
