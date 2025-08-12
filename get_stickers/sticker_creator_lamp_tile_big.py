# sticker_sheet_4x4.py
import argparse
from pathlib import Path
from PIL import Image

# ---------- Page ----------
A4_WIDTH_PX  = 2480
A4_HEIGHT_PX = 3508

# ---------- Layout ----------
MARGIN_X = 20
MARGIN_BOTTOM = 20
HEADER_MARGIN_BOTTOM = 20
PADDING_X = 20
PADDING_Y = 20

COLS = 3
ROWS = 4

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent
EXPORTS    = ROOT / "build" / "exports"
HEADER     = ROOT / "header" / "header.jpeg"

# ---------- Helpers ----------
def load_header_scaled_to_width(path: Path, target_w: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    ar  = img.height / img.width
    return img.resize((target_w, int(target_w * ar)), Image.Resampling.LANCZOS)

def trim_transparent(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA": img = img.convert("RGBA")
    bbox = img.split()[-1].getbbox()
    return img.crop(bbox) if bbox else img

def largest_scale_for_fixed_grid(ref_w, ref_h, cols, rows, work_w, work_h):
    # scan down for snug fit (allow very large if trims are tight)
    for x in range(1200, 14, -1):  # 12.00 -> 0.15
        s = x / 100.0
        sw = int(ref_w * s)
        sh = int(ref_h * s)
        total_w = cols * sw + (cols + 1) * PADDING_X
        total_h = rows * sh + (rows - 1) * PADDING_Y
        if total_w <= work_w and total_h <= work_h:
            return s
    return None

# ---------- Core ----------
def create_sheet(images, header_img: Image.Image, out_path: Path):
    # Trim all first (important for size)
    images = [trim_transparent(im) for im in images]

    canvas = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
    canvas.paste(header_img, (0, 0), header_img)
    header_h = header_img.height

    work_left = MARGIN_X
    work_top  = header_h + HEADER_MARGIN_BOTTOM
    work_w    = A4_WIDTH_PX - 2 * MARGIN_X
    work_h    = A4_HEIGHT_PX - work_top - MARGIN_BOTTOM

    # Use the largest w/h among the four as reference
    ref_w = max(im.width  for im in images)
    ref_h = max(im.height for im in images)

    scale = largest_scale_for_fixed_grid(ref_w, ref_h, COLS, ROWS, work_w, work_h)
    if scale is None:
        raise RuntimeError("Grid doesn't fit. Reduce paddings or header height.")

    scaled = [im.resize((max(1, int(im.width*scale)), max(1, int(im.height*scale))),
                        Image.Resampling.LANCZOS) for im in images]

    sticker_w, sticker_h = scaled[0].size  # after scaling they’re consistent enough
    grid_w = COLS * sticker_w + (COLS + 1) * PADDING_X
    grid_h = ROWS * sticker_h + (ROWS - 1) * PADDING_Y
    grid_left = work_left + max(0, (work_w - grid_w) // 2)
    grid_top  = work_top  + max(0, (work_h - grid_h) // 2)

    # Paste in cyclic order 1,2,3,4,...
    idx = 0
    for r in range(ROWS):
        y = grid_top + r * (sticker_h + PADDING_Y)
        for c in range(COLS):
            x = grid_left + PADDING_X + c * (sticker_w + PADDING_X)
            st = scaled[idx % 4].convert("RGBA")
            canvas.paste(st, (x, y), st)
            idx += 1

    canvas.save(out_path)
    print(f"Saved → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ts", required=True, help="Timestamp suffix, e.g. 20250811_030553")
    ap.add_argument("--prefix", default="fish", help="Prefix, e.g. Lamp / Fish")
    args = ap.parse_args()

    ts = args.ts
    prefix = args.prefix

    names = [f"{prefix}{i}_deformed_{ts}.png" for i in (1,2,3,4)]
    paths = [EXPORTS / n for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {', '.join(p.name for p in missing)} in {EXPORTS}")

    imgs = [Image.open(p).convert("RGBA") for p in paths]
    header_img = load_header_scaled_to_width(HEADER, A4_WIDTH_PX)

    out = EXPORTS / f"sticker_sheet_lamp_{ts}.png"
    create_sheet(imgs, header_img, out)

if __name__ == "__main__":
    main()
