import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
from src.preprocessing import preprocess_image
from src.prediction import predict_digit

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "final_digit_recognition.keras"
LOGO_PATH = BASE_DIR / "assets" / "pu_logo.png"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root{--forest:#063c30;--green:#087653;--mint:#dff5e9;--gold:#d6b66d;--ink:#142b24;--muted:#73847b;--line:#e1ebe5}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 85% 3%,rgba(13,132,91,.10),transparent 25%),radial-gradient(ellipse at 3% 18%,rgba(214,182,109,.12),transparent 23%),#f5f8f6;color:var(--ink)}
.block-container{max-width:1500px;padding-top:3.3rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:linear-gradient(165deg,#052f27,#075b45 70%,#063c30);border-right:1px solid rgba(255,255,255,.12)}
[data-testid="stSidebar"] *{color:#eff9f3!important}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.18)}
[data-testid="stSidebar"] [data-testid="stMetric"]{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);padding:14px;border-radius:16px}
[data-testid="stSidebar"] [data-testid="stMetricLabel"]{color:#c8e2d5!important}
[data-testid="stSidebar"] [data-testid="stMetricValue"]{color:#fff!important}
.university-title{color:var(--forest);font-family:'Manrope',sans-serif;font-size:clamp(20px,2.6vw,32px);line-height:1.1;font-weight:800;letter-spacing:-.7px}
.university-subtitle{color:#48665a;font-size:14px;margin-top:6px;font-weight:800;letter-spacing:1.2px}
.university-dept{color:#84948b;font-size:12px;margin-top:4px}
.hero{position:relative;display:flex;justify-content:space-between;align-items:center;gap:22px;overflow:hidden;padding:42px 42px;border-radius:28px;color:#fff;background:radial-gradient(circle at 82% 50%,rgba(25,160,111,.42),transparent 27%),linear-gradient(115deg,#032d25 0%,#07563f 56%,#087b55 100%);box-shadow:0 22px 55px rgba(4,65,46,.22);margin:22px 0 26px;min-height:250px}
.hero:before{content:"";position:absolute;inset:0;background:linear-gradient(110deg,transparent 55%,rgba(255,255,255,.045));pointer-events:none}
.hero-copy{position:relative;z-index:2;max-width:720px}
.hero-kicker{display:inline-flex;align-items:center;gap:9px;color:#f0d99f;text-transform:uppercase;font-size:10px;letter-spacing:2.4px;font-weight:800;margin-bottom:17px}
.hero-kicker:before{content:"";width:24px;height:2px;background:var(--gold);display:inline-block}
.hero h1{color:white!important;font-family:'Manrope',sans-serif;font-size:clamp(30px,4vw,49px);line-height:1.06;letter-spacing:-1.5px;margin:0 0 14px;font-weight:800}
.hero p{color:#d7eee3;font-size:14px;line-height:1.8;max-width:610px;margin:0}
.hero-pill{display:inline-block;margin-top:22px;border:1px solid rgba(226,199,135,.48);color:#f0d99f;background:rgba(255,255,255,.07);border-radius:30px;padding:8px 14px;font-size:10px;font-weight:800;letter-spacing:1px}
.hero-art{position:relative;z-index:1;flex:0 0 250px;width:250px;height:185px;border:1px solid rgba(255,255,255,.18);border-radius:25px;background:rgba(255,255,255,.07);backdrop-filter:blur(8px);box-shadow:inset 0 1px rgba(255,255,255,.14),0 15px 35px rgba(0,0,0,.12);display:flex;align-items:center;justify-content:center}
.hero-zero{font-family:'Manrope',sans-serif;font-size:128px;font-weight:800;line-height:1;color:#fff;text-shadow:0 0 35px rgba(164,255,209,.28)}
.hero-chip{position:absolute;border:1px solid rgba(255,255,255,.2);background:rgba(3,43,34,.72);color:#e7f8ee;border-radius:11px;padding:7px 11px;font-size:11px;font-weight:800;box-shadow:0 6px 18px rgba(0,0,0,.12)}
.chip-top{top:13px;left:13px}.chip-bottom{right:13px;bottom:13px;color:#f1dca8}
.section-heading{color:var(--forest);font-family:'Manrope',sans-serif;font-size:21px;font-weight:800;margin:8px 0 4px;letter-spacing:-.4px}
.section-caption{color:var(--muted);font-size:13px;margin-bottom:16px}
.info-card,.panel{background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:20px;box-shadow:0 10px 30px rgba(18,58,41,.055)}
.info-card{padding:18px 20px;min-height:98px;position:relative;overflow:hidden}
.info-card:after{content:"";position:absolute;right:-18px;top:-25px;width:76px;height:76px;border-radius:50%;background:rgba(8,118,83,.055)}
.info-label{color:#819187;font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:1.3px;margin-bottom:9px}
.info-value{color:#173b2d;font-size:15px;font-weight:800;position:relative;z-index:1}
.panel{padding:23px;height:100%}
.panel-title{color:var(--forest);font-family:'Manrope',sans-serif;font-size:18px;font-weight:800;margin-bottom:4px;letter-spacing:-.3px}
.panel-subtitle{color:#87958d;font-size:12px;margin-bottom:16px}
.result-banner{background:radial-gradient(circle at 50% 0%,rgba(255,255,255,.8),transparent 55%),linear-gradient(145deg,#e5f7ed,#f8fbf8);border:1px solid #cfe8d9;border-radius:23px;padding:25px;text-align:center;box-shadow:0 12px 28px rgba(12,105,69,.07)}
.result-label{color:#658575;text-transform:uppercase;font-size:10px;letter-spacing:1.7px;font-weight:800}
.result-digit{color:#087653;font-family:'Manrope',sans-serif;font-size:86px;line-height:1.08;font-weight:800;margin:8px 0;text-shadow:0 7px 20px rgba(8,118,83,.12)}
.result-confidence{color:#3e7058;font-size:14px;font-weight:800}
.mini-note{color:#829188;font-size:12px;line-height:1.7}
.footer{text-align:center;color:#84948b;padding:28px 10px 8px;font-size:12px;line-height:1.9}
div.stButton>button{background:linear-gradient(110deg,#075b45,#0c9066);color:white;border:0;border-radius:12px;min-height:46px;font-weight:800;letter-spacing:.15px;box-shadow:0 7px 18px rgba(7,91,69,.19);transition:all .2s ease}
div.stButton>button:hover{background:linear-gradient(110deg,#043d30,#087653);color:white;border:0;transform:translateY(-2px);box-shadow:0 10px 22px rgba(7,91,69,.24)}
.stTabs [data-baseweb="tab-list"]{gap:10px;background:#e8f0eb;padding:6px;border:1px solid #dce8e0;border-radius:14px}
.stTabs [data-baseweb="tab"]{border-radius:10px;color:#567165;font-weight:800;padding:10px 18px}
.stTabs [aria-selected="true"]{background:white!important;color:#075b45!important;box-shadow:0 3px 10px rgba(20,55,40,.09)}
[data-testid="stFileUploader"]{background:#f8fbf9;border:1px dashed #a9c8b7;border-radius:15px;padding:11px}
[data-testid="stImage"] img{border-radius:13px}
hr{border-color:#e2ebe5}
@media(max-width:850px){.hero{padding:30px 25px}.hero-art{flex-basis:175px;width:175px;height:150px}.hero-zero{font-size:100px}}
@media(max-width:650px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:27px 22px;border-radius:20px;min-height:unset}.hero-art{display:none}.panel{padding:17px}.hero h1{font-size:32px}}
</style>
""", unsafe_allow_html=True)

# University identity
header_col1, header_col2 = st.columns([0.8, 5], vertical_alignment="center")
with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=105)
    else:
        st.markdown("<div style='font-size:34px;font-weight:900;color:#075b45;text-align:center'>PU</div>", unsafe_allow_html=True)
with header_col2:
    st.markdown("""
    <div class="university-title">UNIVERSITY OF THE PUNJAB</div>
    <div class="university-subtitle">POTHOHAR CAMPUS, GUJAR KHAN</div>
    <div class="university-dept">Department of Information Technology</div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-copy">
    <div class="hero-kicker">Punjab University · Machine Learning Lab</div>
    <h1>Every digit has<br>a story. We read it.</h1>
    <p>A smart handwritten digit recognition experience powered by a Convolutional Neural Network. Sketch a number, upload your writing, and watch the model decode it.</p>
    <div class="hero-pill">CNN INTELLIGENCE&nbsp;&nbsp; · &nbsp;&nbsp;DIGITS 0—9&nbsp;&nbsp; · &nbsp;&nbsp;INSTANT PREDICTION</div>
  </div>
  <div class="hero-art">
    <div class="hero-chip chip-top">NEURAL VISION</div>
    <div class="hero-zero">8</div>
    <div class="hero-chip chip-bottom">0 — 9 / CLASSIFY</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Project information
st.markdown('<div class="section-heading">Project Overview</div><div class="section-caption">Academic project details</div>', unsafe_allow_html=True)
info1, info2, info3 = st.columns(3)
for column, label, value in [
    (info1, "Submitted By", "Arsalan Ramzan"),
    (info2, "Subject", "Machine Learning"),
    (info3, "Program", "BSIT · 6th Semester")
]:
    with column:
        st.markdown(f'<div class="info-card"><div class="info-label">{label}</div><div class="info-value">{value}</div></div>', unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# Sidebar
st.sidebar.markdown("## PU · Project Dashboard")
st.sidebar.caption("HANDWRITTEN DIGIT RECOGNITION")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.write("**Algorithm:** Custom CNN")
st.sidebar.write("**Dataset:** MNIST")
st.sidebar.write("**Input Shape:** 28 × 28")
st.sidebar.write("**Classes:** 0–9")
st.sidebar.write("**Framework:** TensorFlow / Keras")
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.metric("MNIST Test Accuracy", "99.08%")
st.sidebar.caption("Evaluation performed on the MNIST test dataset.")
st.sidebar.markdown("---")
st.sidebar.markdown("### Technologies")
st.sidebar.write("Python · TensorFlow · NumPy · Streamlit · Pillow")
st.sidebar.markdown("---")
st.sidebar.caption("University of the Punjab\nPothohar Campus, Gujar Khan")

# Prediction helper
def run_prediction(original_image):
    processed = preprocess_image(original_image)
    prediction, confidence, probabilities = predict_digit(model, processed)
    preview = Image.fromarray((processed[0, :, :, 0] * 255).astype(np.uint8))
    st.session_state["prediction"] = prediction
    st.session_state["confidence"] = confidence * 100
    st.session_state["probabilities"] = probabilities
    st.session_state["preview"] = preview

# Main workspace g
st.markdown('<div class="section-heading">Enter the Recognition Studio</div><div class="section-caption">Choose your input method and let the CNN do the reading.</div>', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["✍️  Draw a Digit  ", "📤  Upload an Image"])

with tab1:
    draw_col, guide_col = st.columns([1.15, 0.85], gap="large")
    with draw_col:
        st.markdown('<div class="panel"><div class="panel-title">Drawing Canvas</div><div class="panel-subtitle">Write a single digit from 0 to 9 in the box below.</div>', unsafe_allow_html=True)
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=15,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
            return_image_data=True
        )
        if st.button("Recognize My Digit", type="primary", key="draw_predict", use_container_width=True):
            if canvas_result.image_data is not None and np.min(canvas_result.image_data[:, :, :3]) < 250:
                try:
                    original = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("RGB")
                    run_prediction(original)
                    st.success("Prediction completed!")
                except ValueError as error:
                    st.warning(str(error))
                except Exception as error:
                    st.error(f"Prediction failed: {error}")
            else:
                st.warning("Please draw a digit first.")
        st.markdown('</div>', unsafe_allow_html=True)
    with guide_col:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">How to use</div>
          <div class="panel-subtitle">Follow these simple steps.</div>
          <b>01 · Draw clearly</b><br>Write one digit between 0 and 9. Keep it near the centre.<br><br>
          <b>02 · Use the full canvas</b><br>Make the digit large enough to be recognized.<br><br>
          <b>03 · Run prediction</b><br>Click the green button to send your drawing to the CNN model.<br><br>
          <b>04 · Review the result</b><br>See the predicted digit, confidence score and class probabilities.<hr>
          <div class="mini-note">Tip: Use a simple, continuous stroke and avoid writing multiple digits in one canvas.</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    upload_col, guide_col = st.columns([1.1, 0.9], gap="large")
    with upload_col:
        st.markdown('<div class="panel"><div class="panel-title">Upload Handwritten Digit</div><div class="panel-subtitle">Supported formats: PNG, JPG and JPEG.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], help="Upload an image containing one handwritten digit.")
        if uploaded_file is not None:
            try:
                original = Image.open(uploaded_file).convert("RGB")
                st.image(original, caption="Selected image", use_container_width=True)
                if st.button("Recognize Uploaded Digit", type="primary", key="upload_predict", use_container_width=True):
                    run_prediction(original)
                    st.success("Prediction completed!")
            except ValueError as error:
                st.warning(str(error))
            except Exception as error:
                st.error(f"Image processing failed: {error}")
        else:
            st.markdown('<div class="mini-note">Select a clear image containing one handwritten digit. For best results, use a close-up image with minimal background noise.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with guide_col:
        st.markdown("""
        <div class="panel">
          <div class="panel-title">Image Guidelines</div>
          <div class="panel-subtitle">For more reliable predictions</div>
          • Upload only <b>one handwritten digit</b>.<br><br>
          • Keep the digit clearly visible.<br><br>
          • Avoid shadows and very dark backgrounds.<br><br>
          • Make sure the image is not blurry.<br><br>
          • Blue or black ink on plain paper is recommended.<hr>
          <div class="mini-note">The image is converted into a 28 × 28 grayscale input before being passed to the trained CNN.</div>
        </div>
        """, unsafe_allow_html=True)

# Results
if "prediction" in st.session_state:
    st.markdown("---")
    st.markdown('<div class="section-heading">Recognition Result</div><div class="section-caption">The model has processed your input. Here is the prediction.</div>', unsafe_allow_html=True)
    result_col, detail_col = st.columns([0.8, 1.2], gap="large")
    with result_col:
        st.markdown(f"""
        <div class="result-banner">
          <div class="result-label">Predicted Digit</div>
          <div class="result-digit">{st.session_state["prediction"]}</div>
          <div class="result-confidence">Confidence · {st.session_state["confidence"]:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("MNIST Test Accuracy", "99.08%")
        st.caption("This is the model's test-set accuracy, not the confidence of this individual prediction.")
    with detail_col:
        st.markdown('<div class="panel"><div class="panel-title">Model Input & Probability</div><div class="panel-subtitle">Processed image and confidence distribution across all ten classes.</div>', unsafe_allow_html=True)
        preview_col, chart_col = st.columns([0.7, 1.5], vertical_alignment="center")
        with preview_col:
            st.image(st.session_state["preview"], caption="28 × 28 model input", width=150, clamp=True)
        with chart_col:
            chart_data = {"Digit": list(range(10)), "Probability": st.session_state["probabilities"] * 100}
            st.bar_chart(chart_data, x="Digit", y="Probability", y_label="Probability (%)", x_label="Digit")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Clear Prediction", key="clear_prediction"):
        for key in ["prediction", "confidence", "probabilities", "preview"]:
            st.session_state.pop(key, None)
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
  <b>HANDWRITTEN DIGIT RECOGNITION USING CNN</b><br>
  University of the Punjab · Pothohar Campus, Gujar Khan<br>
  BS Information Technology · 6th Semester<br>
  Developed by Arsalan Ramzan<br>
  Powered by TensorFlow, Keras and Streamlit
</div>
""", unsafe_allow_html=True)
