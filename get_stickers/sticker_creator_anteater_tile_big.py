# sticker_sheet.py
import argparse
from pathlib import Path
from PIL import Image

# ---------- Page ----------
A4_WIDTH_PX  = 2480
A4_HEIGHT_PX = 3508

# ---------- Layout (bigger tiles) ----------
MARGIN_X = 20
MARGIN_BOTTOM = 20
HEADER_MARGIN_BOTTOM = 20
PADDING_X = 40     # tighter gaps
PADDING_Y = 16

COLS = 2
ROWS = 3          # set to 2 for mega tiles

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
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.split()[-1].getbbox()
    return img.crop(bbox) if bbox else img

def largest_scale_for_fixed_grid(ref_w, ref_h, cols, rows, work_w, work_h):
    # scan down for snug fit
    for x in range(1200, 14, -1):  # try up to 12x just in case
        s = x / 100.0
        sw = int(ref_w * s)
        sh = int(ref_h * s)
        total_w = cols * sw + (cols + 1) * PADDING_X
        total_h = rows * sh + (rows - 1) * PADDING_Y
        if total_w <= work_w and total_h <= work_h:
            return s
    return None

# ---------- Core ----------
def create_sheet(img_a: Image.Image, img_b: Image.Image, header_img: Image.Image, out_path: Path):
    canvas = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
    canvas.paste(header_img, (0, 0), header_img)
    header_h = header_img.height

    work_left = MARGIN_X
    work_top  = header_h + HEADER_MARGIN_BOTTOM
    work_w    = A4_WIDTH_PX - 2 * MARGIN_X
    work_h    = A4_HEIGHT_PX - work_top - MARGIN_BOTTOM

    # 🔑 Trim transparent margins before sizing
    img_a = trim_transparent(img_a)
    img_b = trim_transparent(img_b)

    # Use the larger of the two as reference
    ref_w = max(img_a.width,  img_b.width)
    ref_h = max(img_a.height, img_b.height)

    scale = largest_scale_for_fixed_grid(ref_w, ref_h, COLS, ROWS, work_w, work_h)
    if scale is None:
        raise RuntimeError("Grid doesn't fit. Reduce PADDING_X/Y or ROWS/COLS.")

    a_scaled = img_a.resize((max(1, int(img_a.width  * scale)), max(1, int(img_a.height  * scale))), Image.Resampling.LANCZOS)
    b_scaled = img_b.resize((max(1, int(img_b.width  * scale)), max(1, int(img_b.height  * scale))), Image.Resampling.LANCZOS)

    sticker_w, sticker_h = a_scaled.size
    grid_w = COLS * sticker_w + (COLS + 1) * PADDING_X
    grid_h = ROWS * sticker_h + (ROWS - 1) * PADDING_Y
    grid_left = work_left + max(0, (work_w - grid_w) // 2)
    grid_top  = work_top  + max(0, (work_h - grid_h) // 2)

    stickers = [a_scaled, b_scaled]
    idx = 0
    for r in range(ROWS):
        y = grid_top + r * (sticker_h + PADDING_Y)
        for c in range(COLS):
            x = grid_left + PADDING_X + c * (sticker_w + PADDING_X)
            st = stickers[idx % 2].convert("RGBA")
            canvas.paste(st, (x, y), st)
            idx += 1

    canvas.save(out_path)
    print(f"Saved → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ts", required=True, help="Timestamp suffix, e.g. 20250811_034125")
    ap.add_argument("--prefix", default="Anteater", help="Prefix, e.g. Anteater / Lamp / Fish")
    args = ap.parse_args()

    ts = args.ts
    prefix = args.prefix

    img1_path = EXPORTS / f"{prefix}1_deformed_{ts}.png"
    img2_path = EXPORTS / f"{prefix}2_deformed_{ts}.png"
    header_path = HEADER

    img1 = Image.open(img1_path).convert("RGBA")
    img2 = Image.open(img2_path).convert("RGBA")
    header_img = load_header_scaled_to_width(header_path, A4_WIDTH_PX)

    out = EXPORTS / f"sticker_sheet_anteater_{ts}.png"
    create_sheet(img1, img2, header_img, out)

if __name__ == "__main__":
    main()
