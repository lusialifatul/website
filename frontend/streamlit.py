import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:5000/predict"

st.set_page_config(
    page_title="Deteksi Penyakit Daun Jeruk Lemon",
    layout="wide"
)

# =====================
# NAVBAR
# =====================
st.markdown("""
<style>
.navbar {
    background: linear-gradient(90deg,#2ecc71,#27ae60);
    padding: 15px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    font-weight: bold;
    text-align: center;
}
.result-box {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}
.badge {
    padding: 6px 14px;
    border-radius: 20px;
    color: white;
    display: inline-block;
    margin-top: 10px;
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

# =====================
# LAYOUT
# =====================
left, right = st.columns([1,1.2])

with left:
    st.subheader("Upload Gambar")
    uploaded = st.file_uploader(
    label="",
    type=["jpg","png","jpeg"],
    label_visibility="collapsed"
)

with right:
    st.subheader("Hasil Identifikasi")

    if uploaded:
        img = Image.open(uploaded)

        st.image(
            img,
            use_column_width=True,
            caption="Gambar yang diunggah"
        )

        with st.spinner("Menganalisis gambar..."):
            response = requests.post(
                API_URL,
                files={"image": uploaded.getvalue()}
            ).json()

        label = response["label"]
        conf = response["confidence"] * 100
        status = response["status"]

        if status == "not_leaf":
            st.markdown('<span class="badge gray">Bukan Daun Jeruk</span>', unsafe_allow_html=True)
        elif status == "healthy":
            st.markdown('<span class="badge green">Daun Sehat</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge red">Daun Sakit</span>', unsafe_allow_html=True)

        st.markdown(f"### {label}")
        st.progress(int(conf))
        st.write(f"**Akurasi:** {conf:.2f}%")

    else:
        st.info("Silakan unggah gambar untuk memulai identifikasi")
