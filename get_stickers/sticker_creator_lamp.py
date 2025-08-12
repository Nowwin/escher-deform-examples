from PIL import Image, ImageOps
import math

# ===================== Page / I/O =====================
A4_WIDTH_PX  = 2480
A4_HEIGHT_PX = 3508

PATTERN_PATH = "../build/exports/pattern_lamp_deformed_20250811_034324.png"
TILE_PATHS   = [
    "../build/exports/fish1_deformed_20250811_030553.png",
    "../build/exports/fish2_deformed_20250811_030553.png",
    "../build/exports/fish3_deformed_20250811_030553.png",
    "../build/exports/fish4_deformed_20250811_030553.png",
]
HEADER_PATH  = "../header/header.jpeg"
OUTPUT_PATH  = "sticker_sheet_lamp_half_pattern_half_tiles.png"

# ===================== Layout =====================
MARGIN_X = 0
MARGIN_BOTTOM = 0
HEADER_MARGIN_BOTTOM = 30
HEADER_MAX_FRAC = 0.22

SPLIT_FRAC = 0.50            # fraction of space for pattern vs tiles
PATTERN_TILE_GAP = 100        # gap between pattern and tile sections

# Pattern
INNER_PADDING_PATTERN = 0
AUTO_ROTATE = True
FORCE_ROTATE_DEG = None

# Tile grid
COLS = 3
ROWS = 2
PADDING_X = 8
PADDING_Y = 8
EXTRA_CELL_SPACING_X = 20
EXTRA_CELL_SPACING_Y = 20
CELL_INNER_PAD = 40
TILES_TOP_BIAS = 0.15

# ===================== Helpers =====================
def load_header_scaled_to_width(path, page_w, page_h, max_frac=None):
    img = Image.open(path).convert("RGBA")
    ar = img.height / img.width
    w = page_w
    h = int(w * ar)
    if max_frac is not None:
        max_h = int(page_h * max_frac)
        if h > max_h:
            h = max_h
            w = int(h / ar)
    return img.resize((w, h), Image.Resampling.LANCZOS)

def cover(image, box_w, box_h):
    w, h = image.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (max(1, box_w), max(1, box_h)), (0,0,0,0))
    s = max(box_w / w, box_h / h)
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = max(0, (new_w - box_w) // 2)
    top  = max(0, (new_h - box_h) // 2)
    return scaled.crop((left, top, left + box_w, top + box_h))

def rotate_to_min_cover_scale(img, box_w, box_h):
    w, h = img.size
    s0  = max(box_w / w, box_h / h)
    s90 = max(box_w / h, box_h / w)
    return img.rotate(90, expand=True, resample=Image.Resampling.BICUBIC) if s90 < s0 else img

def trim_transparent(img):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.split()[-1].getbbox()
    return img.crop(bbox) if bbox else img

def contain_to_cell(img, cell_w, cell_h, pad_px=0):
    draw_w = max(1, cell_w - 2 * pad_px)
    draw_h = max(1, cell_h - 2 * pad_px)
    w, h = img.size
    s = min(draw_w / w, draw_h / h)
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    result = Image.new("RGBA", (cell_w, cell_h), (0,0,0,0))
    left = pad_px + (draw_w - new_w) // 2
    top  = pad_px + (draw_h - new_h) // 2
    result.paste(scaled, (left, top), scaled)
    return result

# ===================== Main =====================
def main():
    canvas = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")
    header = load_header_scaled_to_width(HEADER_PATH, A4_WIDTH_PX, A4_HEIGHT_PX, HEADER_MAX_FRAC)
    canvas.paste(header, (0, 0), header)
    header_h = header.height

    # Work area
    work_left = MARGIN_X
    work_top  = header_h + HEADER_MARGIN_BOTTOM
    work_w    = A4_WIDTH_PX - 2 * MARGIN_X
    work_h    = A4_HEIGHT_PX - work_top - MARGIN_BOTTOM

    # Split with gap
    pattern_h = max(1, int(work_h * SPLIT_FRAC))
    tiles_h   = max(1, work_h - pattern_h - PATTERN_TILE_GAP)

    # --- Pattern ---
    pattern_box_w = max(1, work_w - 2 * INNER_PADDING_PATTERN)
    pattern_box_h = max(1, pattern_h - 2 * INNER_PADDING_PATTERN)
    pattern = Image.open(PATTERN_PATH).convert("RGBA")
    if FORCE_ROTATE_DEG is not None:
        pattern = pattern.rotate(FORCE_ROTATE_DEG, expand=True, resample=Image.Resampling.BICUBIC)
    elif AUTO_ROTATE:
        pattern = rotate_to_min_cover_scale(pattern, pattern_box_w, pattern_box_h)
    pattern_filled = cover(pattern, pattern_box_w, pattern_box_h)
    px = work_left + (work_w - pattern_filled.width) // 2
    py = work_top + (pattern_h - pattern_filled.height) // 2
    canvas.paste(pattern_filled, (px, py), pattern_filled)

    # --- Tiles ---
    tiles_area_top = work_top + pattern_h + PATTERN_TILE_GAP
    tiles_area_left = work_left
    tiles_area_w = work_w
    tiles_area_h = tiles_h

    tile_imgs = [trim_transparent(Image.open(p).convert("RGBA")) for p in TILE_PATHS]
    total_x_pad = (COLS + 1) * PADDING_X + (COLS - 1) * EXTRA_CELL_SPACING_X
    total_y_pad = (ROWS - 1) * (PADDING_Y + EXTRA_CELL_SPACING_Y)
    cell_w = max(1, (tiles_area_w - total_x_pad) // COLS)
    cell_h = max(1, (tiles_area_h - total_y_pad) // ROWS)

    stickers = [contain_to_cell(im, cell_w, cell_h, pad_px=CELL_INNER_PAD) for im in tile_imgs]

    sticker_w, sticker_h = stickers[0].size
    grid_w = COLS * sticker_w + (COLS + 1) * PADDING_X + (COLS - 1) * EXTRA_CELL_SPACING_X
    grid_h = ROWS * sticker_h + (ROWS - 1) * (PADDING_Y + EXTRA_CELL_SPACING_Y)

    grid_left = tiles_area_left + max(0, (tiles_area_w - grid_w) // 2)
    grid_top  = tiles_area_top + max(0, int((tiles_area_h - grid_h) * TILES_TOP_BIAS))

    idx = 0
    for r in range(ROWS):
        y = grid_top + r * (sticker_h + PADDING_Y + EXTRA_CELL_SPACING_Y)
        for c in range(COLS):
            x = grid_left + PADDING_X + c * (sticker_w + PADDING_X + EXTRA_CELL_SPACING_X)
            st = stickers[idx % len(stickers)]
            canvas.paste(st, (x, y), st)
            idx += 1

    canvas.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
