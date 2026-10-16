import io
import time
import zipfile

import streamlit as st
from PIL import Image, ImageOps
from rembg import new_session

import studio

st.set_page_config(page_title="Product Photo Studio", page_icon="📸", layout="wide")

st.markdown(
    """
<style>
.hero {padding: 1.8rem 2rem; border-radius: 18px; margin-bottom: 1.4rem;
       background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #6d28d9 140%);}
.hero-title {color: #fff; font-size: 2.3rem; font-weight: 800; margin: 0 0 .35rem 0;}
.hero-sub {color: #cbd5e1; font-size: 1.05rem; margin: 0;}
.pill {display: inline-block; padding: .25rem .75rem; margin: .9rem .4rem 0 0;
       border-radius: 999px; background: rgba(255,255,255,.12); color: #e2e8f0; font-size: .8rem;}
.frame-label {font-size: .78rem; font-weight: 700; letter-spacing: .1em;
              text-transform: uppercase; color: #64748b; margin-bottom: .4rem;}
.footer {text-align: center; color: #94a3b8; font-size: .85rem; margin-top: 2.5rem;}
</style>
""",
    unsafe_allow_html=True,
)

MODELS = {"Balanced (U²-Net)": "u2net", "Sharp edges (IS-Net)": "isnet-general-use"}
MAX_BATCH = 10
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


@st.cache_resource(show_spinner=False)
def get_session(model_name: str):
    """Load each AI model once per server and reuse it."""
    return new_session(model_name)


def encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=40)
def cutout_png(data: bytes, model_name: str) -> tuple[bytes, float]:
    """Run the slow AI step once per photo; settings changes reuse the cached cut-out."""
    start = time.perf_counter()
    cut = studio.remove_background(Image.open(io.BytesIO(data)), get_session(model_name))
    return encode(cut, "PNG"), time.perf_counter() - start


# ---------- Sidebar: settings ----------
with st.sidebar:
    st.markdown("### 🎛️ Studio settings")
    preset_name = st.selectbox("Export for", list(studio.PRESETS))
    preset = studio.PRESETS[preset_name]
    st.caption(preset.note)

    backdrop = st.selectbox(
        "Backdrop",
        list(studio.BACKDROPS) + ["Custom colour"],
        disabled=preset.force_white,
        help="Locked to pure white for marketplace main images." if preset.force_white else None,
    )
    custom_color = "#ffffff"
    if backdrop == "Custom colour" and not preset.force_white:
        custom_color = st.color_picker("Backdrop colour", "#f4e9dd")

    fill = st.slider(
        "Product size in frame (%)", 40, 95, int(preset.fill * 100), key=f"fill_{preset_name}"
    ) / 100
    shadow = st.selectbox("Shadow", studio.SHADOWS, index=1)
    enhance = st.toggle("Auto-enhance product", value=True, help="Gentle contrast, colour and sharpness boost.")
    clean = st.toggle("Clean cut-out edges", value=True, help="Removes thin background halos around the product.")

    with st.expander("Advanced"):
        model_label = st.selectbox("AI model", list(MODELS))
        out_fmt = st.radio("File format", ["PNG", "JPEG"], horizontal=True)

model = MODELS[model_label]
ext = "jpg" if out_fmt == "JPEG" else "png"
mime = "image/jpeg" if out_fmt == "JPEG" else "image/png"
settings = dict(
    preset=preset, backdrop=backdrop, custom_color=custom_color,
    fill=fill, shadow=shadow, enhance=enhance, clean_edges=clean,
)

# ---------- Header ----------
st.markdown(
    """
<div class="hero">
  <div class="hero-title">📸 Product Photo Studio</div>
  <p class="hero-sub">Turn casual phone photos into clean, marketplace-ready product shots in seconds.</p>
  <span class="pill">🤖 AI background removal</span>
  <span class="pill">🛒 Marketplace &amp; social presets</span>
  <span class="pill">🌗 Studio shadows &amp; reflections</span>
  <span class="pill">📦 Batch export</span>
</div>
""",
    unsafe_allow_html=True,
)

single_tab, batch_tab, about_tab = st.tabs(["✨ Single photo", "📦 Batch mode", "🧠 How it works"])

# ---------- Single photo ----------
with single_tab:
    up = st.file_uploader("Upload a product photo", type=IMAGE_TYPES, key="single")
    if up is None:
        st.info(
            "👆 Upload a photo to start. Best results: one product, good light, "
            "and a background that contrasts with the product."
        )
    else:
        try:
            with st.spinner("Cutting out your product with AI… (the first run downloads the model)"):
                cut_bytes, secs = cutout_png(up.getvalue(), model)
        except ValueError as err:
            st.error(str(err))
        else:
            cut = Image.open(io.BytesIO(cut_bytes))
            result = studio.compose(cut, **settings)

            before, after = st.columns(2, gap="large")
            with before:
                st.markdown('<div class="frame-label">Before</div>', unsafe_allow_html=True)
                st.image(ImageOps.exif_transpose(Image.open(up)))
            with after:
                st.markdown('<div class="frame-label">After</div>', unsafe_allow_html=True)
                st.image(result)

            m1, m2, m3 = st.columns(3)
            m1.metric("AI cut-out time", f"{secs:.1f} s")
            m2.metric("Output size", f"{result.width} × {result.height}")
            m3.metric("Format", out_fmt)

            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇️ Download studio shot", encode(result, out_fmt),
                file_name=f"studio-shot.{ext}", mime=mime, type="primary",
            )
            d2.download_button(
                "⬇️ Transparent cut-out (PNG)", cut_bytes,
                file_name="cutout.png", mime="image/png",
            )

# ---------- Batch mode ----------
with batch_tab:
    st.write(
        f"Process up to **{MAX_BATCH} photos** at once with the current settings "
        "and download them all as a ZIP."
    )
    files = st.file_uploader(
        "Upload product photos", type=IMAGE_TYPES, accept_multiple_files=True, key="batch"
    )
    if files:
        if len(files) > MAX_BATCH:
            st.warning(f"Only the first {MAX_BATCH} photos will be processed.")
            files = files[:MAX_BATCH]

        if st.button(f"🚀 Process {len(files)} photo(s)", type="primary"):
            progress = st.progress(0.0, text="Starting…")
            zip_buf, previews, failed = io.BytesIO(), [], []
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, f in enumerate(files, 1):
                    progress.progress((i - 1) / len(files), text=f"Processing {f.name} ({i}/{len(files)})…")
                    try:
                        cut_bytes, _ = cutout_png(f.getvalue(), model)
                        img = studio.compose(Image.open(io.BytesIO(cut_bytes)), **settings)
                        stem = f.name.rsplit(".", 1)[0]
                        zf.writestr(f"{i:02d}-{stem}-studio.{ext}", encode(img, out_fmt))
                        thumb = img.copy()
                        thumb.thumbnail((400, 400))
                        previews.append((f.name, thumb))
                    except Exception as err:  # keep going if one photo fails
                        failed.append(f"{f.name}: {err}")
            progress.progress(1.0, text="Done ✅")
            st.session_state["batch"] = {"zip": zip_buf.getvalue(), "previews": previews, "failed": failed}

        batch = st.session_state.get("batch")
        if batch:
            for msg in batch["failed"]:
                st.warning(msg)
            cols = st.columns(5)
            for idx, (name, thumb) in enumerate(batch["previews"]):
                with cols[idx % 5]:
                    st.image(thumb, caption=name)
            st.download_button(
                "⬇️ Download all (ZIP)", batch["zip"],
                file_name="studio-shots.zip", mime="application/zip", type="primary",
            )

# ---------- How it works ----------
with about_tab:
    st.markdown(
        """
#### The pipeline
1. **Orientation fix**: reads the photo's EXIF data so phone photos are never sideways.
2. **AI segmentation**: a U²-Net or IS-Net model (via `rembg`, running on ONNX Runtime) separates the product from the background.
3. **Edge clean-up**: the mask is tightened and softened to remove halos.
4. **Smart framing**: the product is trimmed to its bounds and scaled to fill the right share of the frame for each export preset.
5. **Studio compositing**: backdrop, contact or drop shadow, or a fading floor reflection is added.
6. **Export**: PNG or high-quality JPEG, one at a time or as a ZIP.

#### Engineering notes
- The slow AI step is **cached per photo**, so changing settings re-renders instantly.
- Image logic lives in `studio.py`, separate from the UI, so it can be tested and reused.
- Runs entirely on CPU, with no GPU and no paid APIs.

**Built with** Python · Streamlit · rembg · ONNX Runtime · Pillow
"""
    )

st.markdown(
    '<div class="footer">Built by <a href="https://github.com/jaidheervemula">Jaidheer Vemula</a> · '
    '<a href="https://github.com/jaidheervemula/product-photo-studio">Source on GitHub</a></div>',
    unsafe_allow_html=True,
)
