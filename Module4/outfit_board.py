"""Outfit board: garment crops side by side on white, one labelled tile per piece.

Used by the demo's "Try this swap", the slide figure and (later) the human study's collages, so all
three show outfits the same way. Crops are Module 1's RGBA cut-outs; each is scaled to fit its tile
without distortion. A missing crop draws a grey placeholder instead of failing.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

TILE, PAD, CAPTION_H, BORDER = 200, 12, 28, 4
TITLE_H = 36
HIGHLIGHT = (235, 104, 52)        # #eb6834: the piece being swapped in
PLACEHOLDER = (230, 229, 224)
TEXT = (11, 11, 11)
MAX_CAPTION = 26                  # characters that fit under a 200-px tile


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _fit(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            im.thumbnail((TILE - 2 * BORDER, TILE - 2 * BORDER))
            return im.copy()
    except OSError:     # corrupt or unreadable file (PIL's UnidentifiedImageError is an OSError)
        return None


def outfit_board(tiles, out_path: str, title: str | None = None) -> str:
    """tiles: [(crop_path or None, caption, highlight)]. Writes a PNG and returns out_path."""
    n = max(1, len(tiles))
    top = TITLE_H if title else 0
    board = Image.new("RGB", (PAD + n * (TILE + PAD), top + PAD + TILE + CAPTION_H + PAD), "white")
    draw = ImageDraw.Draw(board)
    font = _font(15)
    if title:
        draw.text((PAD, PAD // 2 + 4), title, fill=TEXT, font=_font(18))
    for i, (path, caption, highlight) in enumerate(tiles):
        x, y = PAD + i * (TILE + PAD), top + PAD
        im = _fit(path)
        if im is None:
            draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], fill=PLACEHOLDER)
            draw.text((x + 10, y + TILE // 2 - 8), "no image", fill=TEXT, font=font)
        else:
            board.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2), im)
        if highlight:
            draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=HIGHLIGHT, width=BORDER)
        draw.text((x, y + TILE + 6), caption[:MAX_CAPTION], fill=TEXT, font=font)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    board.save(out_path)
    return out_path
