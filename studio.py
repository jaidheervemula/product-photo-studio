"""Image-processing core for Product Photo Studio.

Kept free of Streamlit code so it can be tested and reused on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class Preset:
    size: tuple[int, int]
    fill: float  # share of the frame the product may occupy
    force_white: bool = False
    note: str = ""


PRESETS = {
    "Marketplace main image (2000×2000)": Preset(
        (2000, 2000), 0.85, True,
        "Pure white background with the product filling about 85% of the frame, "
        "the main-image style marketplaces like Amazon ask for.",
    ),
    "Shopify / website (2048×2048)": Preset(
        (2048, 2048), 0.78, note="Square image with breathing room for store product pages."
    ),
    "Instagram post (1080×1350)": Preset(
        (1080, 1350), 0.68, note="Portrait 4:5, the format that takes up the most space in the feed."
    ),
    "Instagram story (1080×1920)": Preset(
        (1080, 1920), 0.55, note="Full-screen 9:16 with room for text and stickers."
    ),
    "Square (1024×1024)": Preset((1024, 1024), 0.76, note="General-purpose square image."),
}

SHADOWS = ["None", "Soft contact shadow", "Drop shadow", "Floor reflection"]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise ValueError(f"Not a hex colour: {value!r}")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def _gradient(size, inner, outer, radial=False):
    """Smooth two-colour backdrop: top-to-bottom, or centre-to-edge if radial."""
    grad = Image.radial_gradient("L") if radial else Image.linear_gradient("L")
    mask = grad.resize(size, Image.BICUBIC)
    return Image.composite(Image.new("RGB", size, outer), Image.new("RGB", size, inner), mask)


BACKDROPS = {
    "Pure white": lambda s: Image.new("RGB", s, (255, 255, 255)),
    "Studio grey": lambda s: _gradient(s, (250, 250, 250), (206, 207, 212)),
    "Spotlight": lambda s: _gradient(s, (255, 255, 255), (196, 198, 206), radial=True),
    "Warm beige": lambda s: _gradient(s, (248, 242, 233), (226, 211, 192)),
    "Soft pastel": lambda s: _gradient(s, (240, 243, 255), (252, 233, 243)),
    "Luxury dark": lambda s: _gradient(s, (74, 74, 84), (14, 14, 18), radial=True),
}


def make_backdrop(name: str, size: tuple[int, int], custom_color: str = "#ffffff") -> Image.Image:
    if name == "Custom colour":
        return Image.new("RGB", size, hex_to_rgb(custom_color))
    return BACKDROPS[name](size)


# ---------------------------------------------------------------------------
# Masks and cut-outs
# ---------------------------------------------------------------------------

def prepare_image(image: Image.Image, max_side: int = 2000) -> Image.Image:
    """Fix phone-photo rotation and limit size. All masks are made at this size."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def _fit_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Guard against models that return extra stacked masks or a different size."""
    mask = mask.convert("L")
    if mask.size != size:
        if mask.width == size[0] and mask.height > size[1]:
            mask = mask.crop((0, 0, size[0], size[1]))  # keep the first mask only
        else:
            mask = mask.resize(size, Image.BILINEAR)
    return mask


def ai_mask(image: Image.Image, session) -> Image.Image:
    """Automatic foreground mask from a matting model (U²-Net, IS-Net, BiRefNet)."""
    from rembg import remove

    return _fit_mask(remove(image, session=session, only_mask=True), image.size)


def click_mask(image: Image.Image, base: Image.Image, points, work_side: int = 480) -> Image.Image:
    """Turn the user's clicks into a selection with random-walker segmentation.

    Clicks are seeds: label 1 = keep, 0 = remove. Everything outside the AI mask is
    also seeded as "remove". The random walker spreads the seeds across the photo and
    stops at strong colour edges, which separates touching objects (a book and a mouse).
    For speed it only works on the area around the product and the clicks, downscaled.
    """
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import random_walker

    w, h = image.size
    base = base.convert("L")

    # 1) Crop to the product area plus the clicks: far fewer pixels, much faster.
    bbox = base.point(lambda v: 255 if v > 30 else 0).getbbox() or (0, 0, w, h)
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    pad = int(max(w, h) * 0.03)
    x0 = max(0, min([bbox[0]] + xs) - pad)
    y0 = max(0, min([bbox[1]] + ys) - pad)
    x1 = min(w, max([bbox[2]] + [x + 1 for x in xs]) + pad)
    y1 = min(h, max([bbox[3]] + [y + 1 for y in ys]) + pad)
    crop_img = image.convert("RGB").crop((x0, y0, x1, y1))
    crop_base = base.crop((x0, y0, x1, y1))

    # 2) Downscale the crop for the solver.
    cw, ch = crop_img.size
    s = min(1.0, work_side / max(cw, ch))
    sw, sh = max(1, round(cw * s)), max(1, round(ch * s))
    small = np.asarray(crop_img.resize((sw, sh), Image.BILINEAR), dtype=np.float64) / 255.0
    foreground = np.asarray(crop_base.resize((sw, sh), Image.BILINEAR)) > 30

    # 3) Seeds: background outside the AI mask, plus a small disk at every click.
    seeds = np.zeros((sh, sw), dtype=np.int32)
    seeds[~foreground] = 2
    yy, xx = np.mgrid[0:sh, 0:sw]
    radius = max(3, int(max(sw, sh) * 0.02))
    keep_spots = []
    for x, y, label in points:
        cx, cy = (x - x0) * s, (y - y0) * s
        seeds[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 1 if label == 1 else 2
        if label == 1:
            keep_spots.append((min(sh - 1, max(0, int(cy))), min(sw - 1, max(0, int(cx)))))

    if not (seeds == 1).any():
        return base
    if (seeds == 0).any() and (seeds == 2).any():
        seg = random_walker(small, seeds, beta=130, mode="cg_j", channel_axis=-1)
    else:
        seg = np.where(seeds == 2, 2, 1)
    selected = seg == 1

    # 4) Keep only the regions that actually contain a "keep" click.
    regions, _ = ndimage.label(selected)
    keep_ids = {int(regions[r, c]) for r, c in keep_spots} - {0}
    if keep_ids:
        selected = np.isin(regions, list(keep_ids))

    # 5) Scale back up and place the result into a full-size mask.
    crop_mask = Image.fromarray((selected * 255).astype(np.uint8), mode="L").resize((cw, ch), Image.BILINEAR)
    full = Image.new("L", (w, h), 0)
    full.paste(crop_mask, (x0, y0))
    return full


def combine_masks(base: Image.Image, selection: Image.Image, grow_out: int = 4, grow_in: int = 8) -> Image.Image:
    """Hybrid mask: the selection decides WHAT the product is, the AI mask decides WHERE its edges are.

    Just around the selection's border we keep the AI mask's fine edge; deep inside the
    selection we force full opacity, filling any holes the AI mask left.
    """
    import numpy as np
    from scipy import ndimage

    sel = np.array(selection) > 127
    if not sel.any():
        return base
    near = ndimage.binary_dilation(sel, iterations=grow_out)
    core = ndimage.binary_erosion(sel, iterations=grow_in)
    alpha = np.array(base)
    alpha = np.where(near, alpha, 0)
    alpha = np.where(core, 255, alpha).astype(np.uint8)
    return Image.fromarray(alpha, mode="L")


def to_cutout(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Original colours + mask as alpha (avoids the dark fringes of pre-blended cut-outs)."""
    out = image.convert("RGBA")
    out.putalpha(mask)
    return out


def isolate_main_object(
    cutout: Image.Image,
    min_alpha: int = 40,
    keep_ratio: float = 0.5,
    bump_ratio: float = 0.02,
) -> Image.Image:
    """Keep the main product and drop stray objects, background haze and small bumps.

    1. Erode the mask so objects joined by thin bridges (cables, shadows, haze) separate.
    2. Score each blob by size and closeness to the photo's centre; keep the winner,
       plus any similar-sized blob near the centre (so pairs like earrings survive).
    3. Give every original mask pixel to its nearest blob and keep the chosen blobs' pixels,
       which restores the product's exact outline.
    4. Trim small bumps stuck to the outline (e.g. a dark object under a corner). Bumps
       smaller than `bump_ratio` of the product are removed; larger parts like straps stay.
    """
    import numpy as np
    from scipy import ndimage

    alpha = np.array(cutout.getchannel("A"))
    solid = alpha > 128
    if not solid.any():
        return cutout
    h, w = solid.shape
    structure = np.ones((3, 3), dtype=bool)

    # 1) Break thin bridges.
    radius = max(2, int(max(h, w) * 0.012))
    core = solid
    for r in (radius, radius // 2, 1):
        eroded = ndimage.binary_erosion(solid, structure=structure, iterations=max(1, r))
        if eroded.any():
            core = eroded
            break

    # 2) Pick the main blob(s): big and central wins.
    labels, count = ndimage.label(core)
    if count == 0:
        return cutout
    ids = np.arange(1, count + 1)
    sizes = np.asarray(ndimage.sum(core, labels, ids), dtype=float)
    centres = np.asarray(ndimage.center_of_mass(core, labels, ids), dtype=float).reshape(-1, 2)
    dist = np.hypot((centres[:, 1] - w / 2) / w, (centres[:, 0] - h / 2) / h)
    score = sizes * (1.0 - dist)
    best = int(np.argmax(score))
    keep_ids = [best + 1] + [
        i + 1 for i in range(count)
        if i != best and sizes[i] >= sizes[best] * keep_ratio and dist[i] < 0.35
    ]

    # 3) Restore the exact outline by giving each pixel to its nearest blob.
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    nearest = labels[iy, ix]
    kept = solid & np.isin(nearest, keep_ids)

    # 4) Trim small bumps on the outline.
    bump_r = max(2, int(max(h, w) * 0.015))
    opened = ndimage.binary_opening(kept, structure=structure, iterations=bump_r)
    if opened.any():
        bumps = kept & ~ndimage.binary_dilation(opened, structure=structure, iterations=2)
        bump_labels, n_bumps = ndimage.label(bumps)
        if n_bumps:
            bump_sizes = np.asarray(ndimage.sum(bumps, bump_labels, np.arange(1, n_bumps + 1)))
            small_ids = np.flatnonzero(bump_sizes < kept.sum() * bump_ratio) + 1
            kept &= ~np.isin(bump_labels, small_ids)

    keep = ndimage.binary_dilation(kept, structure=structure, iterations=2)  # keep soft edge pixels
    alpha = np.where(keep & (alpha >= min_alpha), alpha, 0).astype(np.uint8)
    out = cutout.copy()
    out.putalpha(Image.fromarray(alpha))
    return out


def finalize_cutout(cutout: Image.Image, main_only: bool = True) -> Image.Image:
    """Optionally isolate the main object, then trim to the product's bounds."""
    if main_only:
        cutout = isolate_main_object(cutout)
    solid = cutout.getchannel("A").point(lambda v: 255 if v > 20 else 0)
    bbox = solid.getbbox()
    if bbox is None:
        raise ValueError("Couldn't find a product in this photo. Try one with a clearer subject.")
    return cutout.crop(bbox)


def remove_background(image: Image.Image, session, max_side: int = 2000, main_only: bool = True) -> Image.Image:
    """One-call convenience: prepare, mask, cut out and trim."""
    image = prepare_image(image, max_side)
    return finalize_cutout(to_cutout(image, ai_mask(image, session)), main_only)


# ---------------------------------------------------------------------------
# Studio compositing
# ---------------------------------------------------------------------------

def refine_edges(img: Image.Image) -> Image.Image:
    """Shrink the mask by a pixel and soften it to remove background halos."""
    r, g, b, a = img.split()
    a = a.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
    return Image.merge("RGBA", (r, g, b, a))


def enhance_product(img: Image.Image) -> Image.Image:
    alpha = img.getchannel("A")
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    rgb = ImageEnhance.Color(rgb).enhance(1.06)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.25)
    rgb.putalpha(alpha)
    return rgb


def _contact_shadow(canvas, x, y, pw, ph):
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    sw, sh = int(pw * 0.85), max(8, int(min(pw, ph) * 0.08))
    sx, sy = x + (pw - sw) // 2, y + ph - sh // 2
    draw.ellipse([sx, sy, sx + sw, sy + sh], fill=(0, 0, 0, 120))
    blur = max(4, int(canvas.width * 0.012))
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))


def _drop_shadow(canvas, product, x, y):
    w, h = canvas.size
    shadow = Image.new("RGBA", product.size, (0, 0, 0, 0))
    shadow.putalpha(product.getchannel("A").point(lambda v: int(v * 0.35)))
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(shadow, (x + int(w * 0.012), y + int(h * 0.015)))
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(4, int(w * 0.01)))))


def _reflection(canvas, product, x, y):
    top = y + product.height
    refl_h = min(int(product.height * 0.35), canvas.height - top)
    if refl_h <= 0:
        return
    flipped = ImageOps.flip(product).crop((0, 0, product.width, refl_h))
    fade = Image.linear_gradient("L").resize((product.width, refl_h))
    fade = ImageOps.invert(fade).point(lambda v: int(v * 0.30))
    flipped.putalpha(ImageChops.multiply(flipped.getchannel("A"), fade))
    canvas.alpha_composite(flipped, (x, top))


def compose(
    cutout: Image.Image,
    preset: Preset,
    backdrop: str = "Pure white",
    custom_color: str = "#ffffff",
    fill: float | None = None,
    shadow: str = "Soft contact shadow",
    enhance: bool = True,
    clean_edges: bool = True,
) -> Image.Image:
    """Place a cut-out product on a backdrop, framed for the chosen export preset."""
    W, H = preset.size
    fill = preset.fill if fill is None else fill

    product = cutout.convert("RGBA")
    scale = min(W * fill / product.width, H * fill / product.height)
    pw, ph = max(1, round(product.width * scale)), max(1, round(product.height * scale))
    product = product.resize((pw, ph), Image.LANCZOS)
    if clean_edges:
        product = refine_edges(product)
    if enhance:
        product = enhance_product(product)

    x, y = (W - pw) // 2, (H - ph) // 2
    if shadow == "Floor reflection":
        y = max(0, y - int(ph * 0.12))  # lift the product to make room for the reflection

    canvas = make_backdrop("Pure white" if preset.force_white else backdrop, (W, H), custom_color)
    canvas = canvas.convert("RGBA")

    if shadow == "Soft contact shadow":
        _contact_shadow(canvas, x, y, pw, ph)
    elif shadow == "Drop shadow":
        _drop_shadow(canvas, product, x, y)
    elif shadow == "Floor reflection":
        _reflection(canvas, product, x, y)

    canvas.alpha_composite(product, (x, y))
    return canvas.convert("RGB")
