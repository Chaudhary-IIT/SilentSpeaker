# app.py (integrated)
import os, uuid, pathlib
import streamlit as st
from gtts import gTTS
from models import LipReaderModel
from history import init_db, save_history_sqlite, save_history_csv
from translation_tts import translate_text, synthesize_tts

from streamlit_webrtc import webrtc_streamer
import av, cv2, mediapipe as mp

st.set_page_config(page_title="SilentSpeaker", page_icon="🎙️", layout="centered")
BASE = pathlib.Path(__file__).parent.resolve()
UPLOAD_DIR = BASE / "static" / "uploads"
AUDIO_DIR = BASE / "static" / "audio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
init_db()

st.title("🎙️ SilentSpeaker")
st.markdown("##### Read lips. Generate text. Give voice to silence.")

tab_upload, tab_webcam, tab_history = st.tabs(["Upload", "Webcam", "History"])

# -------- Upload flow --------
with tab_upload:
    uploaded_video = st.file_uploader("Upload a speaking video", type=["mp4","mov","avi"])
    tgt_lang = st.selectbox("Target language for TTS", ["Hindi","English","Bengali","Tamil","Telugu"], index=0)
    if uploaded_video:
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}_{uploaded_video.name}"
        with open(fp, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video(str(fp))

        with st.spinner("🔍 Predicting text from lips..."):
            model = getattr(st.session_state, "model", None)
            if model is None:
                st.session_state.model = LipReaderModel()
                model = st.session_state.model
            text_en = model.predict(str(fp))

        st.success("✅ Prediction Complete!")
        st.write(text_en)

        with st.spinner("🌐 Translating..."):
            text_trans = translate_text(text_en, target_lang=tgt_lang)

        st.write(f"Translated → {tgt_lang}: {text_trans}")

        with st.spinner("🔊 Generating TTS..."):
            audio_path = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
            synthesize_tts(text_trans, lang=tgt_lang, out_path=str(audio_path))
        st.audio(str(audio_path), format="audio/mp3")

        # Persist history (choose either or both)
        save_history_sqlite("upload", "en", text_en, text_trans, tgt_lang)
        save_history_csv("upload", "en", text_en, text_trans, tgt_lang)

# -------- Webcam flow (landmarks overlay) --------
with tab_webcam:
    st.caption("Live camera with lip/face contours for ROI debugging")
    MEDIA = {"video": True, "audio": False}

    class LandmarkProcessor:
        def __init__(self):
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            if res.multi_face_landmarks:
                for face_landmarks in res.multi_face_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style()
                    )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="webcam-lips",
        video_processor_factory=LandmarkProcessor,
        media_stream_constraints=MEDIA,
    )

# -------- History view --------
with tab_history:
    st.caption("Saved transcripts (DB and CSV)")
    st.write("• SQLite: static/history.db")
    st.write("• CSV: static/history.csv")
