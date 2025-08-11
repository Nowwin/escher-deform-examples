from PIL import Image

# ===================== Page / I/O =====================
A4_WIDTH_PX  = 2480
A4_HEIGHT_PX = 3508

PATTERN_PATH = "../build/exports/pattern_lamp_deformed_20250811_034324.png"
HEADER_PATH  = "../header/header.jpeg"
OUTPUT_PATH  = "sticker_sheet_lamp_pattern.png"

# ===================== Layout =====================
MARGIN_X = 0               # left/right page margin
MARGIN_BOTTOM = 0          # bottom page margin
HEADER_MARGIN_BOTTOM = 30   # gap under header
HEADER_MAX_FRAC = 0.22      # cap header at this fraction of page height (0.0-1.0)
INNER_PADDING = 0           # set 0 so we truly fill the work area

# Rotation controls
AUTO_ROTATE = True          # choose orientation that requires the least upscaling for cover-fit
FORCE_ROTATE_DEG = None     # set to 90, -90, 180 to force; leave None to use AUTO_ROTATE

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
    """Scale image to completely cover (box_w, box_h), then center-crop."""
    w, h = image.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (max(1, box_w), max(1, box_h)), (0,0,0,0))
    s = max(box_w / w, box_h / h)
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # center-crop to exact box
    left = max(0, (new_w - box_w) // 2)
    top  = max(0, (new_h - box_h) // 2)
    right = left + box_w
    bottom = top + box_h
    return scaled.crop((left, top, right, bottom))

def rotate_to_min_cover_scale(img, box_w, box_h):
    """For cover-fit, pick the orientation that minimizes the needed scale (less upscaling, less crop)."""
    w, h = img.size
    if w == 0 or h == 0:
        return img
    s0  = max(box_w / w, box_h / h)  # scale if not rotated
    s90 = max(box_w / h, box_h / w)  # scale if rotated
    if s90 < s0:
        return img.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    return img

# ===================== Main =====================
def main():
    # Canvas
    canvas = Image.new("RGB", (A4_WIDTH_PX, A4_HEIGHT_PX), "white")

    # Header
    header = load_header_scaled_to_width(HEADER_PATH, A4_WIDTH_PX, A4_HEIGHT_PX, HEADER_MAX_FRAC)
    canvas.paste(header, (0, 0), header)
    header_h = header.height

    # Work area below header (this whole area will be filled)
    work_left = MARGIN_X
    work_top  = header_h + HEADER_MARGIN_BOTTOM
    work_w    = A4_WIDTH_PX - 2 * MARGIN_X
    work_h    = A4_HEIGHT_PX - work_top - MARGIN_BOTTOM

    # Single pattern
    pattern = Image.open(PATTERN_PATH).convert("RGBA")

    # Inner box we will fully fill
    box_w = max(1, work_w - 2 * INNER_PADDING)
    box_h = max(1, work_h - 2 * INNER_PADDING)

    # Rotation choice
    if FORCE_ROTATE_DEG is not None:
        pattern = pattern.rotate(FORCE_ROTATE_DEG, expand=True, resample=Image.Resampling.BICUBIC)
    elif AUTO_ROTATE:
        pattern = rotate_to_min_cover_scale(pattern, box_w, box_h)

    # Cover-fit and paste centered in the work area
    pattern_filled = cover(pattern, box_w, box_h)
    x = work_left + (work_w - pattern_filled.width) // 2
    y = work_top  + (work_h - pattern_filled.height) // 2

    canvas.paste(pattern_filled, (x, y), pattern_filled)
    canvas.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
