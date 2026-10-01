import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Number Plate AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0e1117;
    }

    /* Main content */
    .main {
        padding-top: 1rem;
    }

    /* Title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: #ffffff;
    }

    .subtitle {
        text-align: center;
        color: #a7adb8;
        font-size: 17px;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Information cards */
    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }

    .info-number {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
    }

    .info-label {
        font-size: 14px;
        color: #8b949e;
        margin-top: 5px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 1px dashed #484f58;
        border-radius: 14px;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        height: 48px;
        border: none;
    }

    /* Detection status */
    .success-box {
        background-color: #10261a;
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
        color: #ffffff;
    }

    .warning-box {
        background-color: #2a2110;
        border: 1px solid #9e6a03;
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
        color: #ffffff;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6e7681;
        font-size: 13px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #30363d;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

@st.cache_resource
def load_model():
    return YOLO("Best_Plate_Number_Detecting_Model.pt")


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "detection_result" not in st.session_state:
    st.session_state.detection_result = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "detected_image" not in st.session_state:
    st.session_state.detected_image = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center;">
            <h1 style="color:white;">🚘 Number Plate AI</h1>
            <p style="color:#8b949e;">
                YOLO-powered number plate detection
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.caption(
        f"Current confidence threshold: {confidence:.2f}"
    )

    st.divider()

    st.markdown("### Supported Images")

    st.write("• JPG")
    st.write("• JPEG")
    st.write("• PNG")

    st.divider()

    st.markdown("### Model")

    st.write("YOLO Object Detection")

    st.divider()

    if st.button("🔄 Clear Detection"):

        st.session_state.detection_result = None
        st.session_state.uploaded_image = None
        st.session_state.detected_image = None

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Number Plate Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload an image and use an AI-powered YOLO model
        to detect vehicle number plates.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.markdown(
    '<div class="section-title">Upload Vehicle Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.session_state.uploaded_image = image

    st.markdown(
        '<div class="section-title">Image Preview & Detection</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    with col1:

        st.markdown("### Original Image")

        st.image(
            image,
            use_container_width=True
        )


    # ========================================================
    # DETECTION
    # ========================================================

    with col2:

        st.markdown("### Detection Result")

        detect_button = st.button(
            "🔍 Detect Number Plate",
            type="primary"
        )

        if detect_button:

            with st.spinner(
                "AI model is detecting number plates..."
            ):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )

            result = results[0]

            st.session_state.detection_result = result

            detected_image = result.plot()

            st.session_state.detected_image = detected_image


        # ====================================================
        # SHOW DETECTION RESULT
        # ====================================================

        if st.session_state.detected_image is not None:

            st.image(
                st.session_state.detected_image,
                channels="BGR",
                use_container_width=True
            )

        else:

            st.info(
                "Click the button above to detect number plates."
            )


# ============================================================
# DETECTION INFORMATION
# ============================================================

if st.session_state.detection_result is not None:

    result = st.session_state.detection_result

    number_of_detections = len(result.boxes)


    # ========================================================
    # DETECTION STATISTICS
    # ========================================================

    st.markdown(
        '<div class="section-title">Detection Statistics</div>',
        unsafe_allow_html=True
    )

    if number_of_detections > 0:

        confidences = [
            float(conf)
            for conf in result.boxes.conf
        ]

        average_confidence = (
            sum(confidences) /
            len(confidences)
        )

        highest_confidence = max(
            confidences
        )

        lowest_confidence = min(
            confidences
        )


        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)


        with stat_col1:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-number">
                        {number_of_detections}
                    </div>
                    <div class="info-label">
                        Plates Detected
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with stat_col2:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-number">
                        {average_confidence:.1%}
                    </div>
                    <div class="info-label">
                        Average Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with stat_col3:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-number">
                        {highest_confidence:.1%}
                    </div>
                    <div class="info-label">
                        Highest Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with stat_col4:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-number">
                        {lowest_confidence:.1%}
                    </div>
                    <div class="info-label">
                        Lowest Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.markdown(
            f"""
            <div class="success-box">
                <strong>✓ Number plate detected successfully.</strong>
                <br>
                The model detected {number_of_detections}
                number plate(s) in the uploaded image.
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # DETECTION DETAILS
        # ====================================================

        st.markdown(
            '<div class="section-title">Detection Details</div>',
            unsafe_allow_html=True
        )


        detection_data = []


        for index, box in enumerate(result.boxes):

            confidence_score = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            coordinates = box.xyxy[0].tolist()

            x1, y1, x2, y2 = coordinates


            # Get class name
            if hasattr(result, "names"):

                class_name = result.names.get(
                    class_id,
                    str(class_id)
                )

            else:

                class_name = str(class_id)


            detection_data.append(
                {
                    "Detection": index + 1,
                    "Class": class_name,
                    "Confidence": f"{confidence_score:.2%}",
                    "X1": round(x1, 1),
                    "Y1": round(y1, 1),
                    "X2": round(x2, 1),
                    "Y2": round(y2, 1)
                }
            )


        detection_df = pd.DataFrame(
            detection_data
        )


        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )


    else:

        # ====================================================
        # NO DETECTION
        # ====================================================

        st.markdown(
            """
            <div class="warning-box">
                <strong>⚠ No number plate detected.</strong>
                <br>
                Try another vehicle image or lower the
                confidence threshold from the sidebar.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    if uploaded_file is None:

        st.markdown(
            """
            <div style="
                background:#161b22;
                border:1px solid #30363d;
                border-radius:14px;
                padding:45px;
                text-align:center;
                margin-top:20px;
            ">

                <div style="font-size:50px;">
                    🚘
                </div>

                <h2 style="color:white;">
                    Ready to Detect
                </h2>

                <p style="color:#8b949e;">
                    Upload a vehicle image above to begin
                    number plate detection.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# INFORMATION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">About This Application</div>',
    unsafe_allow_html=True
)

info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.markdown(
        """
        <div class="info-card">

            <div style="font-size:35px;">
                🤖
            </div>

            <h3 style="color:white;">
                YOLO Model
            </h3>

            <p style="color:#8b949e;">
                Uses a trained YOLO object detection
                model to identify vehicle number plates.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info_col2:

    st.markdown(
        """
        <div class="info-card">

            <div style="font-size:35px;">
                🎯
            </div>

            <h3 style="color:white;">
                Confidence Control
            </h3>

            <p style="color:#8b949e;">
                Adjust the confidence threshold to
                control the sensitivity of detection.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with info_col3:

    st.markdown(
        """
        <div class="info-card">

            <div style="font-size:35px;">
                📊
            </div>

            <h3 style="color:white;">
                Detection Analysis
            </h3>

            <p style="color:#8b949e;">
                View confidence scores, bounding boxes,
                coordinates, and detection statistics.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <p>
            Number Plate AI Detection System
        </p>

        <p>
            Built with Streamlit + YOLO
        </p>

    </div>
    """,
    unsafe_allow_html=True
)