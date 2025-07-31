import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Impor streamlit_webrtc
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from aiortc.mediastreams import VideoFrame 

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Deteksi Ganoderma pada Tanaman Kelapa Sawit",
    layout="wide", # Menggunakan layout wide agar konten lebih luas
    initial_sidebar_state="expanded",
)

# --- Gaya Kustom (Nuansa Putih dan Biru Muda) ---
st.markdown(
    """
    <style>

    /* Main app background */
    .stApp {
        background-color: #FFFF; 
        color: #333333; 
    }
    /* Header/Title background */
    .stApp > header {
        background-color: #E0F7FA; 
        padding: 1rem 0;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-bottom: 1px solid #B3E5FC; 
    }
    .stApp > header h1 {
        color: #008CBA; 
        text-align: center;
        margin: 0;
    }

    /* Sidebar background and text */
    .stSidebar {
        background-color: #B3E5FC; 
        color: #000000; 
        padding-top: 2rem;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }
    .stSidebar .stRadio div[role="radiogroup"] label {
        color: #008CBA; 
        font-weight: bold;
    }
    .stSidebar .stRadio div[role="radiogroup"] {
        padding: 10px;
        border-radius: 8px;
        background-color: rgba(255,255,255,0.7); /* Putih agak transparan */
        margin-bottom: 15px;
        border: 1px solid #81D4FA; 
    }
    .stSidebar .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.7); /* Background selectbox di sidebar */
        border-radius: 8px;
        border: 1px solid #81D4FA;
    }

    /* Titles and Headers */
    h1 {
        color: #008CBA; 
        text-align: center;
    }
    h2 {
        color: #008CBA; 
        border-bottom: 2px solid #81D4FA; /
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h3 {
        color: #008CBA;
    }
    /* Buttons */
    .stButton>button {
        background-color: #4FC3F7; 
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease; /* Transisi halus */
    }
    .stButton>button:hover {
        background-color: #81D4FA; /
    }
    /* File Uploader label */
    .stFileUploader label {
        color: #008CBA;
        font-weight: bold;
    }
    /* Info/Warning messages */
    .stAlert {
        background-color: #E0F7FA; 
        color: #006064; 
        border-left: 5px solid #4FC3F7; 
        border-radius: 5px;
        padding: 10px;
    }
    .stAlert.warning {
        background-color: #FFECB3; 
        color: #FF8F00; 
        border-left: 5px solid #FFC107;
        border-radius: 5px;
        padding: 10px;
    }
    /* Text elements */
    p {
        font-size: 16px;
        line-height: 1.6;
    }
    .stRadio {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Inisialisasi Session State Variabel di Awal ---
if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None

# --- Muat Model YOLO ---
@st.cache_resource
def load_model():
    try:
        # PATH MODEL: PASTIKAN INI SESUAI DENGAN LOKASI FILE best.pt ANDA
        model = YOLO('models/best.pt')
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}. Pastikan file 'models/best.pt' ada dan valid.")
        return None

model = load_model()

if model is None:
    st.stop() 

# --- Judul Aplikasi ---
st.title("Deteksi Ganoderma pada Tanaman Kelapa Sawit")

# --- Penjelasan Singkat ---
st.write(
    """
    Jamur **Ganoderma** adalah patogen penyebab **Penyakit Busuk Pangkal Batang (BPB)**,
    ancaman serius bagi perkebunan kelapa sawit yang dapat menyebabkan penurunan produksi hingga kematian tanaman. Aplikasi ini hadir untuk membantu identifikasi cepat jamur Ganoderma pada tanaman kelapa sawit menggunakan model deteksi objek **YOLOv11** yang canggih.
    """
)
st.markdown("---") 

# --- Sidebar ---
st.sidebar.title("Pilih Mode Deteksi")
detection_mode = st.sidebar.radio(
    "Mode Deteksi:",
    ("Unggah Gambar", "Ambil Gambar (Foto)", "Deteksi Realtime")
)

# --- Fungsi Deteksi Gambar (untuk Unggah/Ambil Foto) ---
def detect_on_image(image_input, display_placeholder):
    try:
        # Konversi input menjadi format BGR yang diterima OpenCV dan YOLO
        img_np = None
        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input) # Konversi PIL Image ke NumPy array (RGBA atau RGB)
            if img_np.shape[2] == 4: # Jika RGBA, konversi ke RGB
                img_np = img_np[:, :, :3]
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # Konversi dari RGB ke BGR
        elif isinstance(image_input, np.ndarray):
            # Asumsi jika sudah numpy array, sudah dalam BGR atau akan dihandle oleh YOLO
            img = image_input
        else:
            st.error("Format gambar tidak didukung.")
            return

        # Lakukan deteksi
        results = model(img)

        # Dapatkan frame yang sudah dianotasi (ini akan dalam format BGR)
        annotated_frame = results[0].plot()

        # Tampilkan gambar hasil deteksi (tetap dalam BGR untuk menjaga warna asli)
        display_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

        # Tampilkan hasil deteksi detail
        boxes = results[0].boxes.xyxy.cpu().numpy()
        names = model.names
        if len(boxes) > 0:
            st.subheader("Objek Terdeteksi:")
            found_ganoderma = False
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                conf = results[0].boxes.conf[i].item()
                cls = int(results[0].boxes.cls[i].item())
                label = names[cls]
                st.write(f"- **{label}** (Kepercayaan: {conf:.2f}) pada koordinat: [{x1}, {y1}, {x2}, {y2}]")
                if label == "Ganoderma": # Ganti "Ganoderma" jika nama kelas Anda berbeda
                    found_ganoderma = True
            if found_ganoderma:
                st.warning("Potensi terinfeksi Ganoderma terdeteksi! Segera lakukan penanganan.")
            else:
                st.info("Tidak ada Ganoderma terdeteksi. Namun, selalu lakukan verifikasi visual.")
        else:
            st.info("Tidak ada objek Ganoderma terdeteksi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mendeteksi gambar: {e}")
        st.exception(e)

# --- Konten Utama Berdasarkan Mode Deteksi ---
# --- Unggah Gambar ---
if detection_mode == "Unggah Gambar":
    st.header("Unggah Gambar untuk Deteksi")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Gambar Asli:")
        st.image(image, caption='Gambar Diunggah', use_container_width=True)
        st.subheader("Hasil Deteksi:")
        detection_placeholder = st.empty()
        detect_on_image(image, detection_placeholder)

# --- Ambil Gambar (Foto) ---
elif detection_mode == "Ambil Gambar (Foto)":
    st.header("Ambil Gambar dari Kamera")

    captured_file = st.camera_input("Ambil Foto")

    if captured_file is not None:
        st.session_state.camera_image = Image.open(captured_file)
        st.subheader("Gambar yang Diambil:")
        st.image(st.session_state.camera_image, caption='Gambar dari Kamera', use_container_width=True)
        st.subheader("Hasil Deteksi:")
        detection_placeholder = st.empty()
        detect_on_image(st.session_state.camera_image, detection_placeholder)
    elif st.session_state.camera_image is not None:
        st.info("Ambil foto baru atau lihat hasil sebelumnya.")

# --- Deteksi Realtime (Menggunakan streamlit_webrtc) ---
elif detection_mode == "Deteksi Realtime":
    st.header("Deteksi Realtime dari Webcam")
    st.info("Kamera akan aktif secara otomatis. Berikan izin akses jika diminta.")

    # Class untuk memproses video frame-by-frame
    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self, model):
            self.model = model

        def recv(self, frame: VideoFrame) -> VideoFrame:
            # Konversi WebRTC frame (RGB) ke NumPy array
            img = frame.to_ndarray(format="rgb24") # Input dari webrtc_streamer adalah RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Konversi ke BGR untuk OpenCV/YOLO

            # Lakukan deteksi
            results = self.model(img_bgr, verbose=False) # verbose=False untuk mengurangi log konsol

            # Dapatkan frame yang sudah dianotasi (ini akan dalam format BGR)
            annotated_frame_bgr = results[0].plot()

            # Konversi kembali ke RGB untuk WebRTC output
            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

            return VideoFrame.from_ndarray(annotated_frame_rgb, format="rgb24")


    # Menggunakan webrtc_streamer component
    webrtc_streamer(
        key="ganoderma-detection-webrtc", # Kunci unik untuk instance ini
        mode=WebRtcMode.SENDRECV, # Mengirim video dari browser dan menerima kembali
        rtc_configuration={ # Konfigurasi ICE servers untuk konektivitas
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_processor_factory=lambda: YOLOVideoProcessor(model), # Factory untuk membuat processor
        media_stream_constraints={"video": True, "audio": False}, # Hanya aktifkan video
        async_processing=True, # Proses secara asynchronous agar UI tetap responsif
    )

    st.warning("Webcam akan secara otomatis berhenti saat Anda beralih mode atau menutup tab browser.")