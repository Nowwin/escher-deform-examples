from PIL import Image, ImageOps
import math

# ===================== Page / I/O =====================
A4_WIDTH_PX  = 2480
A4_HEIGHT_PX = 3508
DPI_HINT = 300  # not used directly, just a sanity reminder

IMAGE_PATH_1 = "../build/exports/Anteater1_deformed_20250811_034125.png"
IMAGE_PATH_2 = "../build/exports/Anteater2_deformed_20250811_034125.png"
HEADER_PATH  = "../header/header.jpeg"
OUTPUT_PATH  = "sticker_sheet_anteater_tile.png"

# ===================== Layout =====================
MARGIN_X = 50                 # left/right page margin
MARGIN_BOTTOM = 50            # bottom page margin
HEADER_MARGIN_BOTTOM = 30     # gap under header
HEADER_MAX_FRAC = 0.22        # cap header at this fraction of page height (0.0-1.0). Set None to disable.

# Base gutters between stickers (and left/right inset inside the grid)
PADDING_X = 8
PADDING_Y = 8

# Extra spacing added BETWEEN cells (use this when shapes look cramped due to rotation/diagonals)
EXTRA_CELL_SPACING_X = 20
EXTRA_CELL_SPACING_Y = 20

# Tiny transparent buffer around each sticker after scaling
BLEED_PX = 2

# Force exactly 3x4 grid
COLS = 3
ROWS = 4

# Vertical placement of the grid inside work area: 0.0 = top, 0.5 = centered, 1.0 = bottom
TOP_BIAS = 0.15

# ===================== Scale preferences =====================
BASE_STICKER_SCALE = 0.40
TARGET_AREA_MULTIPLIER = 4.0                      # ~4× area (~2× linear)
MAX_LINEAR_SCALE = BASE_STICKER_SCALE * math.sqrt(TARGET_AREA_MULTIPLIER) * 1.10
MIN_LINEAR_SCALE = 0.15                           # hard floor

# ===================== Helpers =====================
def contain_to_cell(img, cell_w, cell_h, pad_px=40):
    """
    Scale img to *fit* cell (no crop) with optional padding inside.
    Padding is applied on all sides, so actual drawable area is smaller.
    """
    # Ensure padding doesn't make the drawable area negative
    draw_w = max(1, cell_w - 2 * pad_px)
    draw_h = max(1, cell_h - 2 * pad_px)

    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))

    s = min(draw_w / w, draw_h / h)  # contain
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))

    scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create transparent cell and paste image centered with padding
    result = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
    left = pad_px + (draw_w - new_w) // 2
    top = pad_px + (draw_h - new_h) // 2
    result.paste(scaled, (left, top), scaled)
    return result


def load_header_scaled_to_width(path, page_w, page_h, max_frac=None):
    """Loads header, scales to full page width, optionally caps height as fraction of page height."""
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

def trim_transparent(img):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.split()[-1].getbbox()
    return img.crop(bbox) if bbox else img

def add_bleed(img, bleed_px):
    return ImageOps.expand(img, border=bleed_px, fill=(0,0,0,0)) if bleed_px > 0 else img

def analytic_scale_for_grid(work_w, work_h, ref_w, ref_h,
                            cols, rows, pad_x, pad_y,
                            extra_x, extra_y, bleed_px,
                            s_min, s_max):
    """
    Solve for the largest scale that fits both width and height
    when the total grid footprint is:
      width  = cols * (s*ref_w + 2*bleed) + (cols + 1)*pad_x + (cols - 1)*extra_x
      height = rows * (s*ref_h + 2*bleed) + (rows - 1)*(pad_y + extra_y)
    """
    total_x_pad = (cols + 1) * pad_x + (cols - 1) * extra_x + cols * (2 * bleed_px)
    total_y_pad = (rows - 1) * (pad_y + extra_y) + rows * (2 * bleed_px)

    # If ref_w/h were 0 somehow, avoid div-by-zero
    if ref_w <= 0 or ref_h <= 0:
        return s_min

    s_w = (work_w - total_x_pad) / (cols * ref_w)
    s_h = (work_h - total_y_pad) / (rows * ref_h)

    # Largest allowed that fits both
    s = min(s_w, s_h)
    return max(s_min, min(s_max, s))

# ===================== Main layout =====================
def create_sticker_sheet(sticker_raws, header_img):
    canvas = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")

    # Header (trim any transparency, then paste)
    header_img = trim_transparent(header_img)
    canvas.paste(header_img, (0, 0), header_img)
    header_h = header_img.height

    # Work area below header
    work_left = MARGIN_X
    work_top  = header_h + HEADER_MARGIN_BOTTOM
    work_w    = A4_WIDTH_PX - 2 * MARGIN_X
    work_h    = A4_HEIGHT_PX - work_top - MARGIN_BOTTOM

    # Prepare stickers: trim, pick reference as the LARGEST of provided stickers (post-trim)
    processed = [trim_transparent(im.convert("RGBA")) for im in sticker_raws]
    if len(processed) == 0:
        raise RuntimeError("No stickers provided.")
    ref_w = max(2*im.width  for im in processed)
    ref_h = max(2*im.height for im in processed)

    # Analytic scale that fills width/height considering all paddings/extras/bleed
    scale = analytic_scale_for_grid(
        work_w, work_h,
        ref_w, ref_h,
        COLS, ROWS,
        PADDING_X, PADDING_Y,
        EXTRA_CELL_SPACING_X, EXTRA_CELL_SPACING_Y,
        BLEED_PX,
        MIN_LINEAR_SCALE, MAX_LINEAR_SCALE
    )

    # Scale each, then add bleed
    # ----- compute cell box (inner content area, not counting BLEED) -----
    total_x_pad = (COLS + 1) * PADDING_X + (COLS - 1) * EXTRA_CELL_SPACING_X
    total_y_pad = (ROWS - 1) * (PADDING_Y + EXTRA_CELL_SPACING_Y)

    cell_w = max(1, int((work_w - total_x_pad) / COLS))
    cell_h = max(1, int((work_h - total_y_pad) / ROWS))

    # ----- build stickers that *fill* their cells (with center-crop), then add bleed -----
    stickers = []
    for im in processed:
        filled = contain_to_cell(im, cell_w, cell_h)          # cover + crop to cell
        stickers.append(add_bleed(filled, BLEED_PX))        # outer transparent halo

    # final sticker footprint (includes bleed)
    sticker_w, sticker_h = stickers[0].size  # == (cell_w + 2*BLEED_PX, cell_h + 2*BLEED_PX)

    # ----- grid footprint (must match the same spacing math) -----
    grid_w = COLS * sticker_w + (COLS + 1) * PADDING_X + (COLS - 1) * EXTRA_CELL_SPACING_X
    grid_h = ROWS * sticker_h + (ROWS - 1) * (PADDING_Y + EXTRA_CELL_SPACING_Y)

    grid_left = work_left + max(0, (work_w - grid_w) // 2)
    grid_top  = work_top + max(0, int((work_h - grid_h) * TOP_BIAS))

    # ----- paste loop -----
    idx = 0
    for r in range(ROWS):
        y = grid_top + r * (sticker_h + PADDING_Y + EXTRA_CELL_SPACING_Y)
        for c in range(COLS):
            x = grid_left + PADDING_X + c * (sticker_w + PADDING_X + EXTRA_CELL_SPACING_X)
            st = stickers[idx % len(stickers)]
            canvas.paste(st, (x, y), st)
            idx += 1


    return canvas

def main():
    img1 = Image.open(IMAGE_PATH_1).convert("RGBA")
    img2 = Image.open(IMAGE_PATH_2).convert("RGBA")
    header = load_header_scaled_to_width(HEADER_PATH, A4_WIDTH_PX, A4_HEIGHT_PX, HEADER_MAX_FRAC)

    sheet = create_sticker_sheet([img1, img2], header)
    sheet.save(OUTPUT_PATH)
    print(f"Sticker sheet saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
