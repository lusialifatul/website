import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ======================
# CONFIG
# ======================
IMG_SIZE = 224
CONF_THRESHOLD = 0.60

st.set_page_config(
    page_title="Deteksi Penyakit Daun Jeruk Lemon",
    layout="wide"
)

# ======================
# LOAD MODEL (CACHE)
# ======================
@st.cache_resource
def load_all():
    model = load_model("model/mobilenetv3.keras")
    classes = joblib.load("model/label_encoder.pkl")
    return model, classes

model, classes = load_all()

# ======================
# PREPROCESS
# ======================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image).astype("float32")
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

# ======================
# UI STYLE
# ======================
st.markdown("""
<style>
.navbar {
    background: linear-gradient(90deg,#2ecc71,#27ae60);
    padding: 15px;
    border-radius: 14px;
    color: white;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.12);
}
.badge {
    padding: 6px 16px;
    border-radius: 18px;
    color: white;
    font-weight: bold;
    display: inline-block;
    margin-bottom: 10px;
}
.green { background:#2ecc71 }
.red { background:#e74c3c }
.gray { background:#7f8c8d }
</style>

<div class="navbar">
Sistem Deteksi Penyakit Daun Jeruk Lemon
</div>
""", unsafe_allow_html=True)

st.write("")

# ======================
# LAYOUT
# ======================
left, right = st.columns([1,1.3])

with left:
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Unggah Gambar")
    uploaded = st.file_uploader(
    label="",
    type=["jpg","png","jpeg"],
    label_visibility="collapsed"
    )
    # st.markdown("</div>", unsafe_allow_html=True)

with right:
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Hasil Identifikasi")

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)

        with st.spinner("Menganalisis gambar..."):
            img = preprocess_image(image)
            preds = model.predict(img)[0]

        max_conf = float(np.max(preds))
        idx = int(np.argmax(preds))

        if max_conf < CONF_THRESHOLD:
            st.markdown("<span class='badge gray'>Bukan Daun Jeruk</span>", unsafe_allow_html=True)
            st.write("### Objek tidak dikenali sebagai daun jeruk")
        else:
            label = classes[idx]
            status = "Sehat" if "Healthy" in label else "Sakit"

            if status == "Sehat":
                st.markdown("<span class='badge green'>Daun Sehat</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge red'>Daun Sakit</span>", unsafe_allow_html=True)

            st.write(f"### {label}")

        st.progress(int(max_conf * 100))
        st.write(f"**Akurasi:** {max_conf*100:.2f}%")

    else:
        st.info("Silakan unggah gambar untuk memulai prediksi")

    # st.markdown("</div>", unsafe_allow_html=True)
