import base64
import hashlib
import io
import time
import zipfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageOps
from rembg import new_session
from streamlit_image_coordinates import streamlit_image_coordinates

import studio

st.set_page_config(page_title="Product Photo Studio", page_icon="📸", layout="wide")

MODELS = {
    "Balanced (U²-Net)": "u2net",
    "Sharp edges (IS-Net)": "isnet-general-use",
    "Best quality (BiRefNet lite, slower)": "birefnet-general-lite",
}
MAX_BATCH = 10
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
PREVIEW_SIDE = 560  # size of the clickable picker image


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def load_css() -> None:
    css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    css = "\n".join(line for line in css.splitlines() if line.strip())
    html(f"<style>{css}</style>")


def cards(items) -> str:
    parts = [
        f'<div class="step"><div class="num">{num}</div><div class="ico">{icon}</div>'
        f'<div class="t">{title}</div><div class="d">{desc}</div></div>'
        for num, icon, title, desc in items
    ]
    return '<div class="steps">' + "".join(parts) + "</div>"


def encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- AI steps (each cached, so changing settings never re-runs the AI) ----------
@st.cache_resource(show_spinner=False)
def get_session(model_name: str):
    """Load each AI model once per server and reuse it."""
    return new_session(model_name)


@st.cache_data(show_spinner=False, max_entries=20)
def working_png(data: bytes) -> bytes:
    """The upload, rotated correctly and resized. Every mask and click uses this size."""
    return encode(studio.prepare_image(Image.open(io.BytesIO(data))), "PNG")


def load_working(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(working_png(data))).convert("RGB")


@st.cache_data(show_spinner=False, max_entries=20)
def preview_png(data: bytes) -> tuple[bytes, float]:
    """Small copy of the photo for the click picker, plus its scale factor."""
    working = load_working(data)
    preview = working.copy()
    preview.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    return encode(preview, "PNG"), working.width / preview.width


@st.cache_data(show_spinner=False, max_entries=40)
def matte_png(data: bytes, model_name: str) -> tuple[bytes, float]:
    start = time.perf_counter()
    mask = studio.ai_mask(load_working(data), get_session(model_name))
    return encode(mask, "PNG"), time.perf_counter() - start


@st.cache_data(show_spinner=False, max_entries=40)
def click_png(data: bytes, model_name: str, points: tuple) -> bytes:
    mask_bytes, _ = matte_png(data, model_name)
    base = Image.open(io.BytesIO(mask_bytes)).convert("L")
    return encode(studio.click_mask(load_working(data), base, points), "PNG")


@st.cache_data(show_spinner=False, max_entries=40)
def build_cutout(data: bytes, model_name: str, main_only: bool, points: tuple) -> tuple[bytes, float]:
    """Automatic mask, optionally refined by the user's clicks, then trimmed."""
    mask_bytes, secs = matte_png(data, model_name)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    if points:
        selection = Image.open(io.BytesIO(click_png(data, model_name, points))).convert("L")
        mask = studio.combine_masks(mask, selection)
    cutout = studio.to_cutout(load_working(data), mask)
    cutout = studio.finalize_cutout(cutout, main_only=main_only and not points)
    return encode(cutout, "PNG"), secs


def to_data_uri(img: Image.Image, max_side: int = 1000) -> str:
    img = img.convert("RGB").copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------- Cinematic showcase: a virtual camera moving around the result ----------
SHOWCASE_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: transparent; font-family: Inter, -apple-system, system-ui, sans-serif; }
  .scene { position: relative; height: __H__px; border-radius: 28px; overflow: hidden; perspective: 1600px;
           display: flex; align-items: center; justify-content: center; cursor: grab;
           background: radial-gradient(60% 55% at 50% 40%, #1d1d1f 0%, #050505 65%, #000 100%);
           border: 1px solid rgba(255,255,255,.08); animation: sceneIn 1s cubic-bezier(.22,1,.36,1) both; }
  .scene::before { content: ""; position: absolute; width: 140%; height: 60%; top: -22%; left: -20%;
           background: radial-gradient(closest-side, rgba(255,255,255,.10), transparent);
           animation: spot 16s ease-in-out infinite; pointer-events: none; }
  .rig { margin-top: -70px; transform-style: preserve-3d; animation: camera 16s cubic-bezier(.45,0,.2,1) infinite; }
  .tilt { transform-style: preserve-3d; transition: transform .6s cubic-bezier(.22,1,.36,1); }
  .card { position: relative; width: __CW__px; aspect-ratio: __RATIO__; border-radius: 18px; overflow: hidden;
          box-shadow: 0 60px 120px rgba(0,0,0,.95), 0 0 0 1px rgba(255,255,255,.10);
          -webkit-box-reflect: below 16px linear-gradient(transparent 70%, rgba(255,255,255,.16)); }
  .card img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .glare { position: absolute; inset: 0; pointer-events: none;
           background: linear-gradient(115deg, transparent 35%, rgba(255,255,255,.35) 50%, transparent 65%);
           transform: translateX(-130%); animation: glare 16s ease-in-out infinite; }
  .cap { position: absolute; left: 0; right: 0; bottom: 28px; text-align: center; color: #f5f5f7;
         font-size: 22px; font-weight: 700; letter-spacing: -.02em; opacity: 0; pointer-events: none;
         animation: cap 16s ease-in-out infinite; }
  .c2 { animation-delay: 5.33s; } .c3 { animation-delay: 10.67s; }
  .hint { position: absolute; top: 18px; left: 22px; font-size: 11px; font-weight: 600; letter-spacing: .18em;
          text-transform: uppercase; color: #86868b; }
  .live { position: absolute; top: 16px; right: 20px; display: flex; align-items: center; gap: 8px;
          font-size: 11px; font-weight: 700; letter-spacing: .18em; color: #f5f5f7; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: #fff; animation: pulse 1.6s ease-in-out infinite; }
  @keyframes sceneIn { from { opacity: 0; transform: scale(.97); filter: blur(10px); } to { opacity: 1; transform: none; filter: none; } }
  @keyframes camera {
    0%   { transform: translateZ(-140px) rotateY(-34deg) rotateX(10deg) rotateZ(-2deg); }
    18%  { transform: translateZ(20px)   rotateY(-10deg) rotateX(4deg)  rotateZ(0deg); }
    36%  { transform: translateZ(60px)   rotateY(26deg)  rotateX(-5deg) rotateZ(2deg); }
    56%  { transform: translateZ(150px)  rotateY(0deg)   rotateX(0deg)  rotateZ(0deg); }
    76%  { transform: translateZ(10px)   rotateY(38deg)  rotateX(12deg) rotateZ(-3deg); }
    100% { transform: translateZ(-140px) rotateY(-34deg) rotateX(10deg) rotateZ(-2deg); }
  }
  @keyframes glare { 0%, 40% { transform: translateX(-130%); } 56%, 100% { transform: translateX(130%); } }
  @keyframes spot { 0%, 100% { transform: translateX(-10%); } 50% { transform: translateX(25%); } }
  @keyframes cap {
    0% { opacity: 0; transform: translateY(14px); filter: blur(8px); }
    5%, 27% { opacity: 1; transform: none; filter: none; }
    33%, 100% { opacity: 0; transform: translateY(-10px); filter: blur(8px); }
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .25; } }
  @media (prefers-reduced-motion: reduce) {
    .rig, .glare, .cap, .scene, .scene::before, .dot { animation: none !important; }
    .c1 { opacity: 1; }
  }
</style>
<div class="scene" id="scene">
  <div class="hint">Move your cursor to look around</div>
  <div class="live"><span class="dot"></span>LIVE PREVIEW</div>
  <div class="rig" id="rig"><div class="tilt" id="tilt"><div class="card">
    <img src="__IMG__" alt="Studio shot"><div class="glare"></div>
  </div></div></div>
  <div class="cap c1">Studio lighting.</div>
  <div class="cap c2">Every angle.</div>
  <div class="cap c3">Ready to sell.</div>
</div>
<script>
  const scene = document.getElementById("scene"), rig = document.getElementById("rig"), tilt = document.getElementById("tilt");
  scene.addEventListener("pointermove", e => {
    const r = scene.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
    rig.style.animationPlayState = "paused";
    tilt.style.transform = `rotateY(${x * 40}deg) rotateX(${-y * 24}deg) translateZ(40px)`;
  });
  scene.addEventListener("pointerleave", () => {
    tilt.style.transform = "";
    rig.style.animationPlayState = "running";
  });
</script>
"""


def showcase(img: Image.Image, height: int = 560) -> None:
    w, h = img.size
    card_w = min(int(300 * w / h), 520)
    page = (
        SHOWCASE_HTML.replace("__IMG__", to_data_uri(img, 1200))
        .replace("__H__", str(height))
        .replace("__CW__", str(card_w))
        .replace("__RATIO__", f"{w} / {h}")
    )
    components.html(page, height=height + 10)


# ---------- Before / after slider ----------
COMPARE_HTML = """
<style>
  body { margin: 0; padding: 12px 0 20px; background: transparent; font-family: -apple-system, system-ui, sans-serif; }
  .wrap { position: relative; width: min(100%, __MAXW__px); aspect-ratio: __RATIO__; margin: 0 auto;
          border-radius: 22px; overflow: hidden; border: 1px solid rgba(255,255,255,.12);
          box-shadow: 0 30px 80px rgba(0,0,0,.8); animation: rise .9s cubic-bezier(.22,1,.36,1) both; }
  .wrap img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
              user-select: none; pointer-events: none; }
  #after { clip-path: inset(0 0 0 100%); }
  .line { position: absolute; top: 0; bottom: 0; left: 100%; width: 2px; transform: translateX(-50%);
          background: #fff; box-shadow: 0 0 20px rgba(255,255,255,.8); pointer-events: none; }
  .knob { position: absolute; top: 50%; left: 100%; width: 46px; height: 46px; border-radius: 50%;
          transform: translate(-50%, -50%); display: flex; align-items: center; justify-content: center;
          background: #fff; color: #000; font-size: 18px; font-weight: 800; pointer-events: none;
          box-shadow: 0 0 0 6px rgba(255,255,255,.15), 0 10px 30px rgba(0,0,0,.6); }
  .tag { position: absolute; top: 14px; padding: 6px 12px; border-radius: 999px; font-size: 11px; font-weight: 700;
         letter-spacing: .14em; color: #fff; background: rgba(0,0,0,.55); backdrop-filter: blur(8px); pointer-events: none; }
  .tag.l { left: 14px; } .tag.r { right: 14px; }
  input { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: ew-resize; }
  @keyframes rise { from { opacity: 0; transform: translateY(24px) scale(.97); filter: blur(10px); } to { opacity: 1; transform: none; filter: none; } }
  @media (prefers-reduced-motion: reduce) { .wrap { animation: none; } }
</style>
<div class="wrap">
  <img src="__BEFORE__" alt="Before">
  <img id="after" src="__AFTER__" alt="After">
  <div class="line" id="l"></div>
  <div class="knob" id="k">⇆</div>
  <span class="tag l">BEFORE</span><span class="tag r">AFTER</span>
  <input type="range" min="0" max="100" step="0.1" value="100" id="s" aria-label="Compare before and after">
</div>
<script>
  const s = document.getElementById("s"), a = document.getElementById("after"),
        l = document.getElementById("l"), k = document.getElementById("k");
  function set(v) { a.style.clipPath = `inset(0 0 0 ${v}%)`; l.style.left = v + "%"; k.style.left = v + "%"; }
  s.addEventListener("input", e => set(e.target.value));
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (__ANIMATE__ && !reduce) {
    let start = null;
    const sweep = ts => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / 1600, 1);
      const v = 100 - 50 * (1 - Math.pow(1 - p, 3));
      s.value = v; set(v);
      if (p < 1) requestAnimationFrame(sweep);
    };
    setTimeout(() => requestAnimationFrame(sweep), 400);
  } else { s.value = 50; set(50); }
</script>
"""


def compare_slider(before: Image.Image, after: Image.Image, animate: bool, height: int = 520) -> None:
    w, h = after.size
    before = ImageOps.pad(before.convert("RGB"), (w, h), color=(10, 10, 10))
    page = (
        COMPARE_HTML.replace("__BEFORE__", to_data_uri(before))
        .replace("__AFTER__", to_data_uri(after))
        .replace("__RATIO__", f"{w} / {h}")
        .replace("__MAXW__", str(int(height * w / h)))
        .replace("__ANIMATE__", "true" if animate else "false")
    )
    components.html(page, height=height + 40)


# ---------- Click to select ----------
# A fragment reruns on its own: clicking only redraws the picker, not the whole app.
@st.fragment
def point_picker(data: bytes, digest: str) -> None:
    drafts = st.session_state.setdefault("points", {})
    applied_all = st.session_state.setdefault("applied", {})
    pts = drafts.setdefault(digest, [])
    applied = applied_all.get(digest, ())

    with st.expander("🎯 Not quite right? Pick your product", expanded=bool(pts or applied)):
        st.caption(
            "1. Click the product with ➕ Keep.  2. Switch to ➖ Remove and click anything that "
            "should go, like a hand or another object.  3. Press ✨ Apply selection."
        )
        mode = st.radio("Click mode", ["➕ Keep", "➖ Remove"], horizontal=True, key="click_mode")

        preview_bytes, scale = preview_png(data)
        preview = Image.open(io.BytesIO(preview_bytes)).convert("RGB")
        draw = ImageDraw.Draw(preview)
        for x, y, label in pts:
            px, py, r = x / scale, y / scale, 8
            fill, ring = ((255, 255, 255), (0, 0, 0)) if label == 1 else ((0, 0, 0), (255, 255, 255))
            draw.ellipse([px - r, py - r, px + r, py + r], fill=fill, outline=ring, width=3)

        nonce = st.session_state.get("click_nonce", 0)
        click = streamlit_image_coordinates(preview, key=f"click_{digest}_{nonce}")
        if click is not None:
            spot = (click["x"], click["y"], nonce)
            if spot != st.session_state.get("last_click"):
                st.session_state["last_click"] = spot
                label = 1 if mode.startswith("➕") else 0
                pts.append((int(click["x"] * scale), int(click["y"] * scale), label))
                st.rerun(scope="fragment")  # redraw only the picker: instant

        has_keep = any(p[2] == 1 for p in pts)
        changed = tuple(pts) != tuple(applied)
        if pts and not has_keep:
            st.info("Add at least one ➕ Keep point on the product.")
        elif changed and pts:
            st.caption("You have points that aren't applied yet.")

        c1, c2, c3 = st.columns(3)
        if c1.button("✨ Apply selection", type="primary", disabled=not (has_keep and changed), key="apply_points"):
            applied_all[digest] = tuple(pts)
            st.rerun()  # full rerun: rebuild the cut-out with the new selection
        if c2.button("Undo last point", disabled=not pts, key="undo_point"):
            pts.pop()
            st.session_state["click_nonce"] = nonce + 1
            st.rerun(scope="fragment")
        if c3.button("Reset to automatic", disabled=not (pts or applied), key="reset_points"):
            pts.clear()
            applied_all[digest] = ()
            st.session_state["click_nonce"] = nonce + 1
            st.rerun()


load_css()

# ---------- Sidebar: settings ----------
with st.sidebar:
    html('<div class="side-title">Studio settings</div>')
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
    main_only = st.toggle(
        "Keep main object only", value=True,
        help="Removes small extra objects and background haze. Ignored when you pick points yourself.",
    )

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

# ---------- Hero ----------
html(
    '<div class="hero">'
    '<div class="hero-text">'
    '<div class="eyebrow">Product Photo Studio</div>'
    '<div class="headline"><span class="w">Your</span> <span class="w">product.</span><br>'
    '<span class="w">Every</span> <span class="w">angle.</span></div>'
    '<p class="sub">Turn a casual phone photo into a studio-grade, marketplace-ready shot in seconds. '
    'Powered by open-source AI, running on CPU.</p>'
    '<div class="stats"><div><b>6</b><span>Backdrops</span></div>'
    '<div><b>5</b><span>Export presets</span></div>'
    '<div><b>10</b><span>Photos per batch</span></div></div>'
    "</div>"
    '<div class="stage"><div class="device-wrap"><div class="device"><div class="screen">'
    '<div class="island"></div><div class="orb"></div><div class="orb-shadow"></div>'
    '<div class="screen-label">STUDIO</div><div class="glare"></div>'
    "</div></div></div></div>"
    "</div>"
)

ticker = "   ·   ".join([
    "AI background removal", "Click to select", "Marketplace 2000 × 2000", "Instagram 4:5",
    "Floor reflections", "Luxury dark", "Transparent PNG", "Batch export", "Edge clean-up",
])
html(f'<div class="marquee"><div class="track"><span>{ticker}</span><span aria-hidden="true">{ticker}</span></div></div>')

single_tab, batch_tab, about_tab = st.tabs(["Single photo", "Batch mode", "How it works"])

# ---------- Single photo ----------
with single_tab:
    up = st.file_uploader("Upload a product photo", type=IMAGE_TYPES, key="single")
    if up is None:
        html(cards([
            ("01", "⬆️", "Upload", "Any phone photo of a product. Best on a plain surface."),
            ("02", "🤖", "AI cut-out", "The background and stray objects are removed automatically."),
            ("03", "🎯", "Fine-tune", "Click the product, or anything to remove, if the AI needs a hint."),
            ("04", "⬇️", "Export", "Download a marketplace-ready image, or a whole batch as a ZIP."),
        ]))
    else:
        data = up.getvalue()
        digest = hashlib.md5(data).hexdigest()
        applied = st.session_state.setdefault("applied", {}).get(digest, ())
        active = tuple(applied) if any(p[2] == 1 for p in applied) else ()

        cut_bytes = None
        try:
            msg = "Applying your selection…" if active else "Cutting out your product…"
            with st.spinner(msg + " (the first run downloads the AI model)"):
                cut_bytes, secs = build_cutout(data, model, main_only, active)
        except ValueError as err:
            st.error(str(err))

        if cut_bytes:
            cut = Image.open(io.BytesIO(cut_bytes))
            result = studio.compose(cut, **settings)

            animate = st.session_state.get("last_photo") != digest
            st.session_state["last_photo"] = digest

            view = st.radio(
                "View", ["🎬 Showcase", "↔ Before / After"], horizontal=True, label_visibility="collapsed"
            )
            if view.startswith("🎬"):
                showcase(result)
            else:
                compare_slider(load_working(data), result, animate)

            m1, m2, m3 = st.columns(3)
            m1.metric("AI cut-out time", f"{secs:.1f} s")
            m2.metric("Output size", f"{result.width} × {result.height}")
            m3.metric("Selection", f"{len(active)} point(s)" if active else "Automatic")

            d1, d2 = st.columns(2)
            d1.download_button(
                "Download studio shot", encode(result, out_fmt),
                file_name=f"studio-shot.{ext}", mime=mime, type="primary",
            )
            d2.download_button(
                "Transparent cut-out (PNG)", cut_bytes, file_name="cutout.png", mime="image/png",
            )

        point_picker(data, digest)

# ---------- Batch mode ----------
with batch_tab:
    html('<div class="section-title">Your whole catalogue. One click.</div>')
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

        if st.button(f"Process {len(files)} photo(s)", type="primary"):
            progress = st.progress(0.0, text="Starting…")
            zip_buf, previews, failed = io.BytesIO(), [], []
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, f in enumerate(files, 1):
                    progress.progress((i - 1) / len(files), text=f"Processing {f.name} ({i}/{len(files)})…")
                    try:
                        cut_bytes, _ = build_cutout(f.getvalue(), model, main_only, ())
                        img = studio.compose(Image.open(io.BytesIO(cut_bytes)), **settings)
                        stem = f.name.rsplit(".", 1)[0]
                        zf.writestr(f"{i:02d}-{stem}-studio.{ext}", encode(img, out_fmt))
                        thumb = img.copy()
                        thumb.thumbnail((400, 400))
                        previews.append((f.name, thumb))
                    except Exception as err:  # keep going if one photo fails
                        failed.append(f"{f.name}: {err}")
            progress.progress(1.0, text="Done")
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
                "Download all (ZIP)", batch["zip"],
                file_name="studio-shots.zip", mime="application/zip", type="primary",
            )

# ---------- How it works ----------
with about_tab:
    html('<div class="section-title">From phone photo to studio shot.</div>')
    html(cards([
        ("01", "🧭", "Orientation fix", "Reads the photo's EXIF data so phone photos are never sideways."),
        ("02", "🤖", "AI matting", "U²-Net, IS-Net or BiRefNet (via rembg on ONNX Runtime) finds the foreground and its fine edges."),
        ("03", "🎯", "Click to select", "Random-walker segmentation turns your clicks into a selection that follows the photo's edges."),
        ("04", "🧩", "Hybrid mask", "Your selection decides what the product is; the AI mask decides exactly where its edges are."),
        ("05", "📐", "Smart framing", "Stray objects are dropped, then the product is trimmed and scaled for each preset."),
        ("06", "🌗", "Studio compositing", "Backdrop plus a contact shadow, drop shadow or fading floor reflection."),
    ]))
    html('<div class="section-title">Engineering notes.</div>')
    st.markdown(
        "- Every AI step is **cached per photo**, so changing settings re-renders instantly.\n"
        "- The click picker runs as an **isolated fragment**, so points appear instantly without re-running the app.\n"
        "- Click-to-select uses a **lightweight graph algorithm instead of a 350 MB model**, "
        "so it runs comfortably on a free CPU server.\n"
        "- Cut-outs are rebuilt from the **original colours plus the mask**, avoiding dark edge fringes.\n"
        "- Image logic lives in `studio.py`, separate from the UI, so it can be tested and reused."
    )
    html(
        '<div class="section-title">Built with.</div><div>'
        '<span class="chip">Python</span><span class="chip">Streamlit</span>'
        '<span class="chip">rembg</span><span class="chip">ONNX Runtime</span>'
        '<span class="chip">scikit-image</span><span class="chip">SciPy</span>'
        '<span class="chip">Pillow</span></div>'
    )

html(
    '<div class="footer">Built by <a href="https://github.com/jaidheervemula">Jaidheer Vemula</a> · '
    '<a href="https://github.com/jaidheervemula/product-photo-studio">Source on GitHub</a></div>'
)
