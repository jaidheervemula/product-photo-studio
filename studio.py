def isolate_main_object(cutout: Image.Image, min_alpha: int = 40, keep_ratio: float = 0.25) -> Image.Image:
    """Drop stray objects and faint haze, keeping the main product.

    Keeps the largest solid region plus any region at least `keep_ratio` of its size,
    so pairs (e.g. earrings) survive but small extras (a mouse, a cable) are removed.
    """
    import numpy as np
    from scipy import ndimage

    alpha = np.array(cutout.getchannel("A"))
    solid = alpha > 128
    labels, count = ndimage.label(solid)
    if count == 0:
        return cutout
    sizes = ndimage.sum(solid, labels, range(1, count + 1))
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= sizes.max() * keep_ratio]
    keep = np.isin(labels, keep_ids)
    keep = ndimage.binary_dilation(keep, iterations=3)  # keep the soft edge pixels around it
    alpha = np.where(keep & (alpha >= min_alpha), alpha, 0).astype(np.uint8)
    out = cutout.copy()
    out.putalpha(Image.fromarray(alpha))
    return out


def remove_background(image: Image.Image, session, max_side: int = 2000, main_only: bool = True) -> Image.Image:
    """Cut the product out with AI and trim to its bounds. Returns an RGBA image."""
    from rembg import remove  # imported here so the rest of the module stays lightweight

    image = ImageOps.exif_transpose(image).convert("RGBA")  # fix rotated phone photos
    image.thumbnail((max_side, max_side))
    cutout = remove(image, session=session)
    if main_only:
        cutout = isolate_main_object(cutout)
    solid = cutout.getchannel("A").point(lambda v: 255 if v > 20 else 0)
    bbox = solid.getbbox()
    if bbox is None:
        raise ValueError("Couldn't find a product in this photo. Try one with a clearer subject.")
    return cutout.crop(bbox)
