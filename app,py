import io

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
from rembg import new_session, remove

SIDE = 1024  # output image is 1024 x 1024

st.set_page_config(page_title="Product Photo Studio", page_icon="📸", layout="wide")


@st.cache_resource
def get_session():
    """Load the background-removal model once and reuse it."""
    return new_session("u2net")


def hex_to_rgb(value):
    v = value.lstrip("#")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))


def make_background(style, color):
    if style == "Soft grey gradient":
        top, bottom = (250, 250, 250), (210, 210, 212)
        bg = Image.new("RGB", (SIDE, SIDE))
        draw = ImageDraw.Draw(bg)
        for y in range(SIDE):
            t = y / (SIDE - 1)
            c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
            draw.line([(0, y), (SIDE, y)], fill=c)
        return bg
    if style == "Custom colour":
        return Image.new("RGB", (SIDE, SIDE), hex_to_rgb(color))
    return Image.new("RGB", (SIDE, SIDE), (255, 255, 255))


def make_studio_shot(image, style, color, padding, add_shadow):
    image = image.convert("RGBA")
    image.thumbnail((1500, 1500))  # keep processing fast

    cutout = remove(image, session=get_session())
    bbox = cutout.split()[-1].getbbox()  # area where the product is
    if bbox is None:
        raise ValueError("Couldn't find a product in this photo. Try another one.")
    product = cutout.crop(bbox)

    # Scale the product to fit inside the padding
    max_dim = int(SIDE * (1 - 2 * padding / 100))
    scale = max_dim / max(product.size)
    new_w, new_h = int(product.width * scale), int(product.height * scale)
    product = product.resize((new_w, new_h), Image.LANCZOS)
    x, y = (SIDE - new_w) // 2, (SIDE - new_h) // 2

    canvas = make_background(style, color).convert("RGBA")

    if add_shadow:
        shadow = Image.new("RGBA", (SIDE, SIDE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        sw, sh = int(new_w * 0.8), max(12, int(new_h * 0.06))
        sx, sy = (SIDE - sw) // 2, y + new_h - sh // 2
        draw.ellipse([sx, sy, sx + sw, sy + sh], fill=(0, 0, 0, 110))
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))

    canvas.alpha_composite(product, (x, y))
    return canvas.convert("RGB")


st.title("📸 Product Photo Studio")
st.write(
    "Turn a casual product photo into a clean, e-commerce-ready studio shot. "
    "Upload a photo, pick a backdrop, and download the result."
)

left, right = st.columns(2)

with left:
    uploaded = st.file_uploader("Product photo", type=["jpg", "jpeg", "png", "webp"])
    style = st.radio("Backdrop", ["Pure white", "Soft grey gradient", "Custom colour"])
    color = st.color_picker("Custom colour", "#f4e9dd", disabled=style != "Custom colour")
    padding = st.slider("Padding (%)", 5, 30, 12)
    add_shadow = st.checkbox("Add soft shadow", value=True)
    go = st.button("Create studio shot", type="primary", disabled=uploaded is None)
    if uploaded:
        st.image(uploaded, caption="Original")

with right:
    if go and uploaded:
        try:
            with st.spinner("Removing background… (first run downloads the AI model)"):
                result = make_studio_shot(Image.open(uploaded), style, color, padding, add_shadow)
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            st.session_state["result"] = buf.getvalue()
        except ValueError as err:
            st.error(str(err))

    if "result" in st.session_state:
        st.image(st.session_state["result"], caption="Studio shot")
        st.download_button(
            "⬇️ Download PNG",
            data=st.session_state["result"],
            file_name="studio-shot.png",
            mime="image/png",
        )
