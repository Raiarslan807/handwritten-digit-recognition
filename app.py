
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path

from streamlit_drawable_canvas import st_canvas
from src.preprocessing import preprocess_image
from src.prediction import predict_digit


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "final_digit_recognition.keras"
LOGO_PATH = BASE_DIR / "assets" / "pu_logo.png"


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

    .stApp {
        background-color: #f5f7f6;
    }

    .main-header {
        background: linear-gradient(120deg, #064e3b, #047857);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }

    .main-header h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 8px;
    }

    .main-header p {
        color: #e5f5ed;
        font-size: 16px;
        margin: 0;
    }

    .section-card {
        background: white;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #e1e9e4;
        margin-bottom: 15px;
    }

    .university-name {
        color: #065f46;
        font-weight: bold;
        font-size: 19px;
    }

    .project-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .project-value {
        color: #1e293b;
        font-size: 16px;
        font-weight: 600;
    }

    div.stButton > button {
        background-color: #047857;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 10px 20px;
    }

    div.stButton > button:hover {
        background-color: #065f46;
        color: white;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. UNIVERSITY HEADER
# =====================================================

header_col1, header_col2 = st.columns([1, 5])

with header_col1:

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=115)

    else:
        st.markdown(
            "<h1 style='color:#065f46;text-align:center;'>PU</h1>",
            unsafe_allow_html=True
        )

with header_col2:

    st.markdown("""
    <div>
        <div class="university-name">
            UNIVERSITY OF THE PUNJAB
        </div>
        <div style="color:#475569;font-size:15px;">
            POTHOHAR CAMPUS, GUJAR KHAN
        </div>
        <div style="color:#64748b;font-size:13px;">
            Department of Information Technology
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>Handwritten Digit Recognition System</h1>
    <p>
        An Intelligent Handwritten Digit Classification
        System Using Convolutional Neural Network
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 4. PROJECT INFORMATION
# =====================================================

st.subheader("Project Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown("""
    <div class="section-card">
        <div class="project-label">Submitted By</div>
        <div class="project-value">Arsalan Ramzan</div>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="section-card">
        <div class="project-label">Program</div>
        <div class="project-value">BS Information Technology</div>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="section-card">
        <div class="project-label">Semester</div>
        <div class="project-value">6th Semester</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 5. LOAD TRAINED CNN MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


try:

    model = load_model()

    st.sidebar.success("CNN Model Loaded Successfully")

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# =====================================================
# 6. SIDEBAR - MODEL INFORMATION
# =====================================================

st.sidebar.title("Project Dashboard")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("**Algorithm:** Custom CNN")
st.sidebar.write("**Dataset:** MNIST")
st.sidebar.write("**Input Shape:** 28 × 28")
st.sidebar.write("**Classes:** 0–9")
st.sidebar.write("**Framework:** TensorFlow / Keras")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Performance")

st.sidebar.metric(
    "Test Accuracy",
    "99.08%"
)

st.sidebar.caption(
    "Evaluation performed on the MNIST test dataset."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Project Technologies")

st.sidebar.write(
    "- Python\n"
    "- TensorFlow\n"
    "- NumPy\n"
    "- Streamlit\n"
    "- Pillow"
)


# =====================================================
# 7. PREDICTION HELPER
# =====================================================

def run_prediction(original_image):

    processed = preprocess_image(original_image)

    prediction, confidence, probabilities = predict_digit(
        model,
        processed
    )

    preview = Image.fromarray(
        (processed[0, :, :, 0] * 255).astype(np.uint8)
    )

    st.session_state["prediction"] = prediction
    st.session_state["confidence"] = confidence * 100
    st.session_state["probabilities"] = probabilities
    st.session_state["preview"] = preview


# =====================================================
# 8. MAIN APPLICATION
# =====================================================

st.markdown("---")

st.subheader("Digit Recognition")

st.write(
    "Draw a digit between 0 and 9 or upload an image "
    "of a handwritten digit."
)

tab1, tab2 = st.tabs([
    "Draw a Digit",
    "Upload an Image"
])


# =====================================================
# TAB 1 - DRAW DIGIT
# =====================================================

with tab1:

    st.markdown("### Draw Your Digit")

    col1, col2 = st.columns([1, 1])

    with col1:

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

    with col2:

        st.markdown("### Instructions")

        st.write(
            "1. Draw one digit from 0 to 9.\n\n"
            "2. Keep the digit inside the canvas.\n\n"
            "3. Click the prediction button.\n\n"
            "4. View the predicted digit and confidence."
        )

        if st.button(
            "Predict Drawn Digit",
            type="primary",
            key="draw_predict"
        ):

            if (
                canvas_result.image_data is not None
                and np.max(canvas_result.image_data[:, :, :3]) > 0
            ):

                try:

                    original = Image.fromarray(
                        canvas_result.image_data.astype("uint8")
                    ).convert("RGB")

                    run_prediction(original)

                    st.success("Prediction completed!")

                except ValueError as error:

                    st.warning(str(error))

                except Exception as error:

                    st.error(f"Prediction failed: {error}")

            else:

                st.warning("Please draw a digit first.")


# =====================================================
# TAB 2 - UPLOAD IMAGE
# =====================================================

with tab2:

    st.markdown("### Upload Handwritten Digit")

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image containing one handwritten digit."
    )

    if uploaded_file is not None:

        try:

            original = Image.open(uploaded_file).convert("RGB")

            st.image(
                original,
                caption="Uploaded Handwritten Digit",
                width=250
            )

            if st.button(
                "Predict Uploaded Digit",
                type="primary",
                key="upload_predict"
            ):

                run_prediction(original)

                st.success("Prediction completed!")

        except ValueError as error:

            st.warning(str(error))

        except Exception as error:

            st.error(f"Image processing failed: {error}")


# =====================================================
# 9. DISPLAY PREDICTION RESULTS
# =====================================================

if "prediction" in st.session_state:

    st.markdown("---")

    st.header("Prediction Results")

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Predicted Digit",
            st.session_state["prediction"]
        )

    with result2:

        st.metric(
            "Confidence Score",
            f"{st.session_state['confidence']:.2f}%"
        )

    with result3:

        st.metric(
            "MNIST Test Accuracy",
            "99.08%"
        )

    st.markdown("---")

    preview_col, chart_col = st.columns([1, 2])

    with preview_col:

        st.subheader("Processed Input")

        st.image(
            st.session_state["preview"],
            width=180,
            clamp=True
        )

        st.caption(
            "28 × 28 grayscale image supplied to the CNN."
        )

    with chart_col:

        st.subheader("Class Probability Distribution")

        chart_data = {
            "Digit": list(range(10)),
            "Probability": (
                st.session_state["probabilities"] * 100
            )
        }

        st.bar_chart(
            chart_data,
            x="Digit",
            y="Probability"
        )

    if st.button("Clear Prediction"):

        for key in [
            "prediction",
            "confidence",
            "probabilities",
            "preview"
        ]:
            st.session_state.pop(key, None)

        st.rerun()


# =====================================================
# 10. PROJECT FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div class="footer">
    <b>Handwritten Digit Recognition Using CNN</b><br>
    University of the Punjab | Pothohar Campus, Gujar Khan<br>
    BS Information Technology | 6th Semester<br>
    Developed by Arsalan Ramzan<br><br>
    Powered by TensorFlow, Keras and Streamlit
</div>
""", unsafe_allow_html=True)
