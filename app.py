# BONE FRACTURE DETECTION STUDIO
#
#  CNN + PSO OPTIMIZED MODEL (cnn_pso_model.h5)
#  UPLOAD X-RAY IMAGE
#  DETECTION WITH BOUNDING BOX + CONFIDENCE SCORE
#  PROFESSIONAL CYBERPUNK UI (matches DSP Voice Studio style)
#
#  FIX APPLIED:
#  Keras alphabetically assigns class indices:
#    {'fractured': 0, 'normal': 1}
#  So sigmoid output = P(class 1) = P(Normal)
#  Therefore P(Fractured) = 1.0 - model_output
#  All prediction logic updated accordingly.

import os
import io
import warnings
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import cv2

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:
    HAS_TF = False

# ____________________________________________________________ PAGE CONFIG

st.set_page_config(
    page_title="Bone Fracture Detection — CNN+PSO",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ____________________________________________________________ CSS STYLES (Cyberpunk Theme)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,200,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(255,50,100,0.04) 0%, transparent 60%),
        linear-gradient(160deg, #020408 0%, #05090f 50%, #020408 100%);
}

.main .block-container { max-width: 1400px; padding-top: 0.5rem; }

/* HERO */
.hero {
    text-align: center;
    padding: 18px 0 28px;
    border-bottom: 1px solid rgba(0,200,255,0.12);
    margin-bottom: 24px;
}
.hero h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #00c8ff, #00ffe0, #ff3264);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero sub {
    color: #4a7a7a;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 3px;
}

/* GLASS PANEL */
.glass {
    background: rgba(5, 15, 25, 0.75);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 0 40px rgba(0,200,255,0.03), inset 0 1px 0 rgba(0,200,255,0.05);
}

/* SECTION TITLE */
.sec-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    letter-spacing: 2px;
    color: #00c8ff;
    border-left: 3px solid #00c8ff;
    padding-left: 12px;
    margin-bottom: 16px;
}

/* METRIC BOX */
.metric-box {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.15);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    color: #00c8ff;
    font-size: 1.5rem;
    font-weight: 700;
}
.metric-label { color: #4a7a7a; font-size: 0.75rem; letter-spacing: 1px; }

/* BADGES */
.badge-cnn {
    display: inline-block;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.35);
    color: #00c8ff;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-right: 8px;
    font-family: 'Share Tech Mono', monospace;
}
.badge-pso {
    display: inline-block;
    background: rgba(255,50,100,0.1);
    border: 1px solid rgba(255,50,100,0.35);
    color: #ff3264;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    font-family: 'Share Tech Mono', monospace;
}
.badge-ok {
    display: inline-block;
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.35);
    color: #00ff88;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1px;
    font-family: 'Share Tech Mono', monospace;
}

/* BUTTONS */
.stButton > button {
    width: 100%;
    border-radius: 10px !important;
    border: 1px solid rgba(0,200,255,0.3) !important;
    padding: 12px !important;
    color: #00c8ff !important;
    font-weight: 700 !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 2px !important;
    font-size: 1rem !important;
    background: rgba(0,200,255,0.07) !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.1);
    transition: 0.25s ease;
}
.stButton > button:hover {
    background: rgba(0,200,255,0.15) !important;
    box-shadow: 0 0 30px rgba(0,200,255,0.2) !important;
}

section[data-testid="stSidebar"] {
    background: #020408;
    border-right: 1px solid rgba(0,200,255,0.08);
}

/* RESULT BANNERS */
.result-fracture {
    background: linear-gradient(135deg, rgba(255,50,100,0.12), rgba(255,100,0,0.08));
    border: 1px solid rgba(255,50,100,0.4);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-normal {
    background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,200,255,0.08));
    border: 1px solid rgba(0,255,136,0.35);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 3px;
}
.result-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ____________________________________________________________ CONSTANTS

DARK_BG  = '#05090f'
PLOT_BG  = '#080d14'
TEXT_COL = '#c8d8d8'
GRID_COL = '#0d1a20'
CYAN     = '#00c8ff'
RED      = '#ff3264'
GREEN    = '#00ff88'
ORANGE   = '#ffa040'

IMG_SIZE   = (224, 224)
MODEL_PATH = "cnn_pso_model.h5"

# ____________________________________________________________ HELPERS

def style_ax(ax):
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.title.set_color(CYAN)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor('#0d2020')
    ax.grid(alpha=0.12, color=GRID_COL, linewidth=0.7)

def new_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(DARK_BG)
    return fig, ax

# ____________________________________________________________ MODEL LOADING

@st.cache_resource
def load_model(path):
    if not HAS_TF:
        return None
    if not os.path.exists(path):
        return None
    try:
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

# ____________________________________________________________ PREPROCESSING

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0), img

# ____________________________________________________________ DETECTION + DRAWING

def draw_detection(pil_img, label, confidence, color_rgb):
    """Draw a detection rectangle and label on the image."""
    img_draw = pil_img.copy().resize((480, 480))
    draw = ImageDraw.Draw(img_draw)

    W, H = img_draw.size

    # Margin: ~10% inset for the detection box
    margin = int(W * 0.08)
    box = [margin, margin, W - margin, H - margin]

    # Draw thick border
    border_color = color_rgb
    for offset in range(4):
        draw.rectangle(
            [box[0]-offset, box[1]-offset, box[2]+offset, box[3]+offset],
            outline=border_color
        )

    # Corner accents
    corner_len = 30
    thick = 5
    corners = [
        (box[0], box[1], box[0]+corner_len, box[1]),
        (box[0], box[1], box[0], box[1]+corner_len),
        (box[2]-corner_len, box[1], box[2], box[1]),
        (box[2], box[1], box[2], box[1]+corner_len),
        (box[0], box[3]-corner_len, box[0], box[3]),
        (box[0], box[3], box[0]+corner_len, box[3]),
        (box[2]-corner_len, box[3], box[2], box[3]),
        (box[2], box[3]-corner_len, box[2], box[3]),
    ]
    for c in corners:
        draw.line(c, fill=border_color, width=thick)

    # Label banner
    banner_h = 42
    draw.rectangle([box[0]-3, box[1]-banner_h-3, box[2]+3, box[1]-3], fill=(10, 15, 25, 210))

    # Text
    label_text = f"  {label.upper()}  |  {confidence*100:.1f}%"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    draw.text((box[0]+8, box[1]-banner_h+8), label_text, fill=border_color, font=font)

    return img_draw

# ____________________________________________________________ PLOTS

def plot_confidence_bar(prob_fractured):
    prob_normal = 1.0 - prob_fractured
    labels = ["Not Fractured", "Fractured"]
    values = [prob_normal * 100, prob_fractured * 100]
    colors = [GREEN, RED]

    fig, ax = new_fig(figsize=(7, 3))
    bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.45)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", color=TEXT_COL)
    ax.set_title("Model Confidence Scores", color=CYAN)
    for bar, val in zip(bars, values):
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va='center', color=TEXT_COL,
                fontsize=10, fontfamily='monospace')
    style_ax(ax)
    plt.tight_layout(pad=0.5)
    return fig

def plot_image_channels(img_arr):
    """Show RGB channel histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    fig.patch.set_facecolor(DARK_BG)
    ch_colors = [RED, GREEN, CYAN]
    ch_names  = ["Red", "Green", "Blue"]
    for i, (ax, col, name) in enumerate(zip(axes, ch_colors, ch_names)):
        ax.hist(img_arr[:,:,i].flatten(), bins=64, color=col, alpha=0.8)
        ax.set_title(f"{name} Channel", color=col)
        style_ax(ax)
    plt.tight_layout(pad=0.5)
    return fig

def plot_grayscale_analysis(img_arr):
    gray = np.mean(img_arr, axis=2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor(DARK_BG)

    axes[0].imshow(gray, cmap='bone')
    axes[0].set_title("Grayscale View", color=CYAN, fontsize=10)
    axes[0].axis('off')
    axes[0].set_facecolor(PLOT_BG)

    axes[1].hist(gray.flatten(), bins=80, color=CYAN, alpha=0.75)
    axes[1].set_title("Pixel Intensity Distribution", color=CYAN, fontsize=10)
    style_ax(axes[1])

    plt.tight_layout(pad=0.5)
    return fig

# ____________________________________________________________ SESSION STATE

if 'result' not in st.session_state:
    st.session_state.result = None

# ____________________________________________________________ SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0;border-bottom:1px solid rgba(0,200,255,0.1);'>
        <div style='font-size:2.5rem;'>🦴</div>
        <div style='font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;
                    color:#00c8ff;letter-spacing:3px;'>FRACTURE AI</div>
        <div style='font-family:Share Tech Mono,monospace;color:#4a7a7a;
                    font-size:0.72rem;letter-spacing:2px;'>CNN + PSO EDITION v1.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='padding:14px 0 8px;'>
    <div style='color:#00c8ff;font-family:Rajdhani,sans-serif;font-weight:600;
                letter-spacing:2px;margin-bottom:10px;'>MODULES</div>
    </div>
    <div style='font-size:0.82rem;color:#4a8a8a;line-height:2;font-family:Share Tech Mono,monospace;'>
    ◈ CNN Image Classifier<br>
    ◈ PSO-Optimized Weights<br>
    ◈ Detection Bounding Box<br>
    ◈ Confidence Score Display<br>
    ◈ RGB Channel Analysis<br>
    ◈ Grayscale Inspection<br>
    ◈ Download Result Image
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Rajdhani,sans-serif;color:#00c8ff;letter-spacing:2px;
                font-weight:600;margin-bottom:8px;'>MODEL INFO</div>
    <table style='width:100%;font-size:0.75rem;color:#4a8a8a;
                  font-family:Share Tech Mono,monospace;border-collapse:collapse;'>
    <tr><td style='color:#7a9a9a;padding:3px;'>Architecture</td>
        <td style='color:#00c8ff;'>CNN (4 Conv layers)</td></tr>
    <tr><td style='color:#7a9a9a;padding:3px;'>Optimizer</td>
        <td style='color:#ff3264;'>PSO + AdamW</td></tr>
    <tr><td style='color:#7a9a9a;padding:3px;'>Input Size</td>
        <td style='color:#00ff88;'>224 × 224 × 3</td></tr>
    <tr><td style='color:#7a9a9a;padding:3px;'>Output</td>
        <td style='color:#ffa040;'>Binary (Sigmoid)</td></tr>
    <tr><td style='color:#7a9a9a;padding:3px;'>Classes</td>
        <td style='color:#00ff88;'>Fractured / Normal</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model path input
    st.markdown("""
    <div style='font-family:Rajdhani,sans-serif;color:#00c8ff;letter-spacing:2px;
                font-weight:600;margin-bottom:8px;'>MODEL PATH</div>
    """, unsafe_allow_html=True)
    model_path_input = st.text_input("", value=MODEL_PATH,
                                     label_visibility="collapsed",
                                     help="Path to your .h5 model file")

    if not HAS_TF:
        st.warning("⚠ TensorFlow not installed.\nDetection disabled.\n`pip install tensorflow`")

# ____________________________________________________________ HEADER

st.markdown("""
<div class='hero'>
    <h1>🦴 BONE FRACTURE DETECTION</h1>
    <sub>CNN + PSO OPTIMIZER  ·  X-RAY ANALYSIS  ·  REAL-TIME DETECTION</sub>
</div>
""", unsafe_allow_html=True)

# ____________________________________________________________ MAIN CONTENT

col_upload, col_result = st.columns([1, 1], gap="large")

# ── LEFT: UPLOAD & SETTINGS ───────────────────────────────
with col_upload:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>UPLOAD X-RAY IMAGE</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='color:#4a7a7a;font-size:0.83rem;margin-bottom:14px;'>
    Upload a bone X-ray image. The CNN+PSO model will analyze it and detect
    whether a fracture is present, with a confidence score and bounding box overlay.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload X-ray (JPG / PNG / JPEG)",
        type=["jpg", "jpeg", "png"],
        help="Supports standard X-ray image formats"
    )

    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded X-Ray", use_container_width=True)

        w, h = pil_img.size
        mode = pil_img.mode

        for col, val, lbl in zip(
            st.columns(3),
            [f"{w}×{h}", mode, f"{uploaded_file.size // 1024} KB"],
            ["Resolution", "Color Mode", "File Size"]
        ):
            with col:
                st.markdown(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-label'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── DETECTION SETTINGS ────────────────────────────────
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>DETECTION SETTINGS</div>", unsafe_allow_html=True)

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probability threshold above which the result is classified as Fractured"
    )
    st.markdown(
        f"<span class='badge-cnn'>CNN</span> Threshold: "
        f"<span style='font-family:Share Tech Mono,monospace;color:#ff3264;'>{threshold:.2f}</span> "
        f"&nbsp;|&nbsp; <span class='badge-pso'>PSO</span> Optimized LR + Dropout",
        unsafe_allow_html=True
    )

    show_analysis = st.checkbox("Show Image Analysis (channels + histogram)", value=True)

    btn_detect = st.button("🔬  RUN FRACTURE DETECTION", key="btn_detect")
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT: RESULTS ────────────────────────────────────────
with col_result:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>DETECTION RESULT</div>", unsafe_allow_html=True)

    if btn_detect:
        if uploaded_file is None:
            st.warning("Upload an X-ray image first.")
        else:
            # Load model
            model = load_model(model_path_input)

            if not HAS_TF:
                st.error("TensorFlow not installed. Cannot run detection.")
            elif model is None:
                st.error(
                    f"Model file not found: `{model_path_input}`\n\n"
                    "Make sure `cnn_pso_model.h5` is in the same directory as this script "
                    "and the path is correct."
                )
            else:
                with st.spinner("Running CNN+PSO inference..."):
                    pil_img_fresh = Image.open(uploaded_file)
                    input_arr, resized_img = preprocess_image(pil_img_fresh)
                    raw_output = float(model.predict(input_arr, verbose=0)[0][0])

                # ──────────────────────────────────────────────────────────────
                # FIX: Keras flow_from_directory sorts classes alphabetically:
                #   class 0 = "fractured"   (f comes before n)
                #   class 1 = "normal"
                # Sigmoid output = P(class 1) = P(Normal)
                # Therefore: P(Fractured) = 1.0 - raw_output
                # ──────────────────────────────────────────────────────────────
                prob_fractured = 1.0 - raw_output   # ← THE FIX

                is_fractured = prob_fractured >= threshold
                label    = "Fractured" if is_fractured else "Not Fractured"
                conf     = prob_fractured if is_fractured else (1.0 - prob_fractured)
                box_rgb  = (255, 50, 100) if is_fractured else (0, 255, 136)
                css_cls  = "result-fracture" if is_fractured else "result-normal"
                icon     = "⚠️" if is_fractured else "✅"
                txt_col  = "#ff3264" if is_fractured else "#00ff88"

                st.session_state.result = {
                    "label": label,
                    "prob_fractured": prob_fractured,
                    "conf": conf,
                    "is_fractured": is_fractured,
                    "raw_output": raw_output
                }

                # Result banner
                st.markdown(f"""
                <div class='{css_cls}'>
                    <div style='color:#4a7a7a;font-family:Share Tech Mono,monospace;
                                font-size:0.8rem;letter-spacing:2px;margin-bottom:6px;'>
                        DIAGNOSIS RESULT
                    </div>
                    <div class='result-label' style='color:{txt_col};'>{icon} {label.upper()}</div>
                    <div class='result-conf' style='color:{txt_col};'>
                        Confidence: {conf*100:.1f}%
                    </div>
                    <div style='color:#4a7a7a;font-family:Share Tech Mono,monospace;
                                font-size:0.75rem;margin-top:8px;'>
                        P(Fractured): {prob_fractured:.4f} &nbsp;|&nbsp; Raw model output: {raw_output:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Annotated image with detection box
                pil_img_box = Image.open(uploaded_file)
                annotated = draw_detection(pil_img_box, label, conf, box_rgb)
                st.image(annotated, caption=f"Detection: {label}", use_container_width=True)

                # Save annotated image for download
                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                buf.seek(0)
                st.download_button(
                    "⬇  Download Annotated Result (PNG)",
                    buf,
                    file_name=f"fracture_detection_{label.replace(' ','_')}.png",
                    mime="image/png"
                )

    elif st.session_state.result is None:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#2a4a4a;'>
            <div style='font-size:4rem;margin-bottom:16px;'>🦴</div>
            <div style='font-family:Rajdhani,sans-serif;font-size:1.3rem;
                        letter-spacing:2px;margin-bottom:8px;'>AWAITING IMAGE</div>
            <div style='font-family:Share Tech Mono,monospace;font-size:0.8rem;'>
                Upload an X-ray and click DETECT
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ____________________________________________________________ CONFIDENCE CHART + ANALYSIS

if btn_detect and uploaded_file and st.session_state.result:
    res = st.session_state.result

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>CONFIDENCE ANALYSIS</div>", unsafe_allow_html=True)
    # Pass prob_fractured so the bar chart always shows the correct orientation
    st.pyplot(plot_confidence_bar(res["prob_fractured"]))
    st.markdown("</div>", unsafe_allow_html=True)

    if show_analysis:
        pil_img_ana = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
        img_arr_ana = np.array(pil_img_ana) / 255.0

        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>IMAGE ANALYSIS</div>", unsafe_allow_html=True)
        st.pyplot(plot_image_channels(img_arr_ana))
        st.pyplot(plot_grayscale_analysis(img_arr_ana))
        st.markdown("</div>", unsafe_allow_html=True)

# ____________________________________________________________ MODEL ARCHITECTURE INFO

with st.expander("📋  CNN + PSO Architecture Details"):
    st.markdown("""
    <div style='font-family:Share Tech Mono,monospace;font-size:0.82rem;
                color:#4a8a8a;line-height:2;'>
    <b style='color:#00c8ff;'>Input:</b> 224 × 224 × 3 (RGB X-ray image, normalized 0–1)<br>
    <b style='color:#00c8ff;'>Conv2D(32, 3×3)</b> → ReLU → MaxPool(2×2)<br>
    <b style='color:#00c8ff;'>Conv2D(64, 3×3)</b> → ReLU → MaxPool(2×2)<br>
    <b style='color:#00c8ff;'>Conv2D(128, 3×3)</b> → ReLU → MaxPool(2×2)<br>
    <b style='color:#00c8ff;'>Conv2D(64, 3×3)</b> → ReLU → MaxPool(2×2)<br>
    <b style='color:#00c8ff;'>Flatten</b> → Dense(128, ReLU) →
    <b style='color:#ff3264;'>Dropout(PSO-optimized)</b><br>
    <b style='color:#00ff88;'>Dense(1, Sigmoid)</b> — Binary output (Fractured / Not Fractured)<br><br>
    <b style='color:#ffa040;'>PSO Optimization:</b> Particle Swarm Optimizer searched the
    hyperparameter space [LR: 1e-5 → 1e-3, Dropout: 0.2 → 0.5] using 8 particles × 5 iterations,
    minimizing validation loss. Final weights trained with AdamW for 15 epochs.<br><br>
    <b style='color:#00c8ff;'>Class Index Note:</b> Keras flow_from_directory assigns indices
    alphabetically → fractured=0, normal=1. Sigmoid output = P(normal), so
    P(fractured) = 1 − model_output. This correction is applied automatically.
    </div>
    """, unsafe_allow_html=True)

# ____________________________________________________________ FOOTER

st.markdown("""
<div style='text-align:center;color:#1a3030;margin-top:40px;
            font-family:Share Tech Mono,monospace;font-size:0.75rem;
            border-top:1px solid #0a1a1a;padding-top:18px;letter-spacing:2px;'>
BONE FRACTURE DETECTION STUDIO  ·  CNN + PSO OPTIMIZER  ·  TENSORFLOW / KERAS  ·  v1.1
</div>
""", unsafe_allow_html=True)