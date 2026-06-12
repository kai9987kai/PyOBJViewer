"""Generate assets/icon.ico — an isometric cube on a rounded dark tile.

Run from the repo root:  py packaging/make_icon.py
Requires Pillow (build-time only; the app itself does not need it).
"""
from pathlib import Path

from PIL import Image, ImageDraw

BASE = 512
TILE = "#1c2735"
TOP = "#8fb0e8"
LEFT = "#4a72c4"
RIGHT = "#2d4f96"
EDGE = "#cfe0f8"


def draw_icon() -> Image.Image:
    img = Image.new("RGBA", (BASE, BASE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    radius = BASE * 0.22
    draw.rounded_rectangle((16, 16, BASE - 16, BASE - 16), radius=radius, fill=TILE)

    # Isometric cube: center and half-extents tuned for the tile.
    cx, cy = BASE / 2, BASE / 2 + 14
    w = BASE * 0.30   # half-width of the iso diamond
    h = BASE * 0.17   # half-height of the top diamond
    d = BASE * 0.30   # vertical drop of the side faces

    top = [(cx, cy - 2 * h), (cx + w, cy - h), (cx, cy), (cx - w, cy - h)]
    left = [(cx - w, cy - h), (cx, cy), (cx, cy + d), (cx - w, cy - h + d)]
    right = [(cx + w, cy - h), (cx, cy), (cx, cy + d), (cx + w, cy - h + d)]

    draw.polygon(left, fill=LEFT)
    draw.polygon(right, fill=RIGHT)
    draw.polygon(top, fill=TOP)

    width = max(3, BASE // 64)
    for face in (left, right, top):
        draw.line(face + [face[0]], fill=EDGE, width=width, joint="curve")

    return img


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"
    out.parent.mkdir(exist_ok=True)
    icon = draw_icon()
    icon.save(
        out,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
