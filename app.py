import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import io

# --- Page Configuration ---
st.set_page_config(page_title="ParkVision", layout="wide")

# ── Inject custom CSS matching ParkVision dark theme ──────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
  }

  .stApp {
    background-color: #0d1117 !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Top Header Bar ── */
  .pv-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 28px 0;
    border-bottom: 1px solid #1e2d3d;
    margin-bottom: 28px;
  }
  .pv-logo-box {
    width: 42px; height: 42px;
    background: #1b3a5c;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 14px; font-weight: 700;
    color: #58a6ff;
    letter-spacing: 1px;
    border: 1px solid #2d5f8a;
  }
  .pv-title-group { display: flex; flex-direction: column; }
  .pv-title {
    font-size: 28px; font-weight: 700; color: #e6edf3;
    letter-spacing: 0.5px; line-height: 1;
  }
  .pv-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px; color: #8b949e;
    letter-spacing: 3px; margin-top: 4px;
  }

  /* ── Section Labels ── */
  .pv-section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px; color: #8b949e;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px dashed #21262d;
  }

  /* ── Stat Cards ── */
  .pv-stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 24px 0;
  }
  .pv-stat-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
  }
  .pv-stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
  }
  .pv-stat-card.total::before  { background: #58a6ff; }
  .pv-stat-card.empty::before  { background: #3fb950; }
  .pv-stat-card.occupied::before { background: #d29922; }
  .pv-stat-card.rate::before   { background: #f78166; }

  .pv-stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px; color: #8b949e;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 8px;
  }
  .pv-stat-value {
    font-size: 36px; font-weight: 700; line-height: 1;
  }
  .pv-stat-card.total  .pv-stat-value { color: #58a6ff; }
  .pv-stat-card.empty  .pv-stat-value { color: #3fb950; }
  .pv-stat-card.occupied .pv-stat-value { color: #d29922; }
  .pv-stat-card.rate   .pv-stat-value { color: #e6c07b; }

  .pv-stat-sub {
    font-size: 13px; color: #6e7681;
    margin-top: 5px; letter-spacing: 0.5px;
  }

  /* ── Occupancy Bar ── */
  .pv-bar-wrap {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 20px;
  }
  .pv-bar-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }
  .pv-bar-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px; color: #8b949e; letter-spacing: 2px;
  }
  .pv-bar-pct {
    font-family: 'Share Tech Mono', monospace;
    font-size: 15px; color: #d29922; font-weight: 700;
  }
  .pv-bar-track {
    background: #21262d; border-radius: 2px; height: 4px; width: 100%;
  }
  .pv-bar-fill {
    background: linear-gradient(90deg, #d29922, #e3b341);
    border-radius: 2px; height: 4px;
    transition: width 0.8s ease;
  }

  /* ── Image panels ── */
  .pv-panel-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px; color: #58a6ff;
    letter-spacing: 2px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
  }
  .pv-panel-label::before {
    content: '';
    width: 6px; height: 6px;
    background: #58a6ff; border-radius: 50%;
    display: inline-block;
  }

  /* ── File uploader reskin ── */
  [data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 6px !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #58a6ff !important;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    color: #8b949e !important;
  }

  /* ── Run Detection Button ── */
  .stButton > button {
    background: #1b3a5c !important;
    color: #58a6ff !important;
    border: 1px solid #2d5f8a !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    padding: 10px 24px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #2d5f8a !important;
    border-color: #58a6ff !important;
    color: #cae8ff !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d !important;
  }
  [data-testid="stSidebar"] .stTextInput input,
  [data-testid="stSidebar"] .stSlider {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
  [data-testid="stSidebar"] .stMarkdown {
    color: #8b949e !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
  }
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #c9d1d9 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1px !important;
    font-size: 16px !important;
  }

  /* Slider track & thumb */
  [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #f47067 !important;
    border-color: #f47067 !important;
  }
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {
    background: #f47067 !important;
  }

  /* ── Metrics override ── */
  [data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 14px !important;
  }
  [data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important; color: #8b949e !important;
    letter-spacing: 2px !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 32px !important; font-weight: 700 !important;
    color: #58a6ff !important;
  }

  /* ── Download button ── */
  [data-testid="stDownloadButton"] > button {
    background: #161b22 !important;
    color: #3fb950 !important;
    border: 1px solid #238636 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
  }
  [data-testid="stDownloadButton"] > button:hover {
    background: #1a3626 !important;
  }

  /* ── Info / warning boxes ── */
  .stAlert {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #8b949e !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 4px !important;
  [data-testid="column"] { gap: 0 !important; }

  /* progress bar */
  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #d29922, #e3b341) !important;
  }
  [data-testid="stProgressBar"] {
    background: #21262d !important;
    border-radius: 2px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pv-header">
  <div class="pv-logo-box">PV</div>
  <div class="pv-title-group">
    <div class="pv-title">ParkVision</div>
    <div class="pv-subtitle">PARKING SPACE DETECTION SYSTEM</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Class config ───────────────────────────────────────────────────────────────
CLASS_CONFIG = {
    "empty":    {"color_hex": "#3fb950", "color_bgr": (80, 185, 63),  "emoji": "🟢"},
    "occupied": {"color_hex": "#d29922", "color_bgr": (34, 153, 210), "emoji": "🟠"},
}
FALLBACK_COLOR_BGR = (88, 166, 255)

CANDIDATE_PATHS = [
    "best.pt",
    "final_model/best.pt",
    "runs/detect/v8n_aug_2/weights/best.pt",
]

def find_default_model():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return "yolov8n.pt"

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("### ⚙ Model Settings")
model_name = st.sidebar.text_input(
    "YOLO Weight File",
    value=find_default_model(),
    help=(
        "Auto-detected in this order:\n"
        "1. best.pt\n"
        "2. final_model/best.pt\n"
        "3. runs/detect/v8n_aug_2/weights/best.pt"
    ),
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model `{model_path}`: {e}")
        return None

model = load_model(model_name)

if model:
    st.sidebar.success(f"✅ `{model_name}`")
    if hasattr(model, "names") and model.names:
        st.sidebar.caption("Classes: " + ", ".join(model.names.values()))
else:
    st.sidebar.error("⚠️ No model loaded.")

# ── Input Section ──────────────────────────────────────────────────────────────
st.markdown('<div class="pv-section-label">INPUT</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a parking lot image here — JPG · JPEG · PNG",
    type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"],
    label_visibility="visible",
)

# ── Detection Pipeline ─────────────────────────────────────────────────────────
if uploaded_file is not None:
    image     = Image.open(uploaded_file)
    img_array = np.array(image.convert("RGB"))

    if True:   # auto-run on upload
        st.markdown("---")
        st.markdown('<div class="pv-section-label">VISUAL ANALYSIS</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown('<div class="pv-panel-label">ORIGINAL INPUT</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        if model:
            with st.spinner("Scanning parking spaces..."):
                results = model(img_array, conf=confidence_threshold)

                counts  = {name: 0 for name in CLASS_CONFIG}
                n_other = 0

                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cls_id     = int(box.cls[0].cpu().numpy())
                        conf       = float(box.conf[0].cpu().numpy())
                        class_name = r.names[cls_id].lower() if r.names else str(cls_id)

                        if class_name in CLASS_CONFIG:
                            cfg   = CLASS_CONFIG[class_name]
                            color = cfg["color_bgr"]
                            label = f"{class_name.upper()} {conf:.2f}"
                            counts[class_name] += 1
                        else:
                            color = FALLBACK_COLOR_BGR
                            label = f"{class_name} {conf:.2f}"
                            n_other += 1

                        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_y = max(y1 - 18, th + 4)
                        cv2.rectangle(img_array, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
                        cv2.putText(img_array, label, (x1 + 2, label_y - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 17, 23), 1)

            with col2:
                st.markdown('<div class="pv-panel-label">DETECTION OUTPUT</div>', unsafe_allow_html=True)
                st.image(img_array, use_container_width=True)

            # ── Stats ──────────────────────────────────────────────────────────
            n_empty    = counts["empty"]
            n_occupied = counts["occupied"]
            total      = n_empty + n_occupied + n_other
            occupancy  = (n_occupied / total * 100) if total > 0 else 0

            st.markdown('<div class="pv-section-label" style="margin-top:28px">DETECTION SUMMARY</div>',
                        unsafe_allow_html=True)

            pct_str = f"{occupancy:.1f}%"
            empty_sub   = f"{(n_empty/total*100):.0f}% of total" if total > 0 else "—"
            occ_sub     = f"{(n_occupied/total*100):.0f}% of total" if total > 0 else "—"

            st.markdown(f"""
            <div class="pv-stats-row">
              <div class="pv-stat-card total">
                <div class="pv-stat-label">TOTAL SPACES</div>
                <div class="pv-stat-value">{total}</div>
                <div class="pv-stat-sub">detected boxes</div>
              </div>
              <div class="pv-stat-card empty">
                <div class="pv-stat-label">AVAILABLE</div>
                <div class="pv-stat-value">{n_empty}</div>
                <div class="pv-stat-sub">empty spaces</div>
              </div>
              <div class="pv-stat-card occupied">
                <div class="pv-stat-label">OCCUPIED</div>
                <div class="pv-stat-value">{n_occupied}</div>
                <div class="pv-stat-sub">with vehicle</div>
              </div>
              <div class="pv-stat-card rate">
                <div class="pv-stat-label">OCCUPANCY %</div>
                <div class="pv-stat-value">{pct_str}</div>
                <div class="pv-stat-sub">of total capacity</div>
              </div>
            </div>

            <div class="pv-bar-wrap">
              <div class="pv-bar-header">
                <span class="pv-bar-title">OCCUPANCY RATE</span>
                <span class="pv-bar-pct">{pct_str}</span>
              </div>
              <div class="pv-bar-track">
                <div class="pv-bar-fill" style="width:{occupancy:.1f}%"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Download ───────────────────────────────────────────────────────
            buf = io.BytesIO()
            Image.fromarray(img_array).save(buf, format="PNG")
            st.download_button(
                label="⬇  DOWNLOAD ANNOTATED IMAGE",
                data=buf.getvalue(),
                file_name="parkvision_result.png",
                mime="image/png",
            )

        else:
            with col2:
                st.warning("Model not loaded. Check the YOLO weight file path in the sidebar.")

else:
    st.markdown("""
    <div style="
      background:#161b22; border:1px dashed #30363d; border-radius:6px;
      padding:28px; text-align:center; margin-top:12px;
      font-family:'Share Tech Mono',monospace; font-size:12px; color:#6e7681;
      letter-spacing:1px;
    ">
      Upload a parking lot image above to begin detection
    </div>
    """, unsafe_allow_html=True)