# app.py (enhanced with Grad-CAM and explainability)
import os, uuid, pathlib, itertools
import streamlit as st
from gtts import gTTS
from models import LipReaderModel, make_gradcam_heatmap, overlay_heatmap
from history import init_db, save_history_sqlite, save_history_csv
from translation_tts import translate_text, synthesize_tts
import sqlite3
from streamlit_webrtc import webrtc_streamer
import av, cv2, mediapipe as mp
import numpy as np
import tensorflow as tf
from PIL import Image

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

# Use a cached singleton for your model instance
@st.cache_resource
def get_model():
    return LipReaderModel()

# -------- Upload flow --------
with tab_upload:
    uploaded_video = st.file_uploader("Upload a speaking video", type=["mp4", "mov", "avi"])
    tgt_lang = st.selectbox("Target language for TTS", ["Hindi", "English", "Bengali", "Tamil", "Telugu"], index=0)

    if uploaded_video:
        fp = UPLOAD_DIR / f"{uuid.uuid4().hex}_{uploaded_video.name}"
        with open(fp, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video(str(fp))

        model = get_model()

        with st.spinner("🔍 Predicting text from lips..."):
            text_en = model.predict(str(fp))

        st.success("✅ Prediction Complete!")
        st.write(text_en)

        # Show model summary if user wants
        if st.checkbox("Show Model Architecture Summary"):
            st.text("LipReaderModel Architecture:")
            buffer = []
            model.model.summary(print_fn=lambda x: buffer.append(x))
            st.text("\n".join(buffer))

        # Grad-CAM visualization
        if st.checkbox("Show Grad-CAM Heatmap for lip frames"):
            st.markdown("### Grad-CAM Heatmaps: Highlighting model attention on lips")
            # Extract frames and preprocess again
            frames = model.extract_lip_frames(str(fp))
            preprocessed = model.preprocess_frames(frames)
            # Expand dims & convert to tensor for TF
            input_tensor = tf.convert_to_tensor(np.expand_dims(preprocessed, axis=0), dtype=tf.float32)

            ## predictions: model output after forward pass on input_tensor
            predictions = model.model(input_tensor)  # shape: (batch_size, time, num_classes)

            # Choose the class with the highest average score across time frames (scalar)
            avg_predictions = tf.reduce_mean(predictions, axis=1)  # (batch, num_classes)
            pred_index = tf.argmax(avg_predictions[0])  # scalar int - predicted class for batch 0

            heatmap = make_gradcam_heatmap(
                input_tensor, model.model, last_conv_layer_name="conv3d", pred_index=pred_index
            )

            # Assuming heatmap shape matches frames, show first few
            heatmaps = [heatmap[i] for i in range(min(5, heatmap.shape[0]))]


            # Show heatmap overlays alongside original frames
            cols = st.columns(min(5, len(heatmaps)))
            for idx, col in enumerate(cols):
                orig_frame = np.uint8(frames[idx] * 255)
                orig_frame_color = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2BGR)
                overlayed_img = overlay_heatmap(orig_frame_color, heatmaps[idx])
                img_pil = Image.fromarray(overlayed_img)
                col.image(img_pil, caption=f"Frame {idx + 1} Grad-CAM", use_column_width=True)

        with st.spinner("🌐 Translating..."):
            text_trans = translate_text(text_en, target_lang=tgt_lang)

        st.write(f"Translated → {tgt_lang}: {text_trans}")

        with st.spinner("🔊 Generating TTS..."):
            audio_path = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
            synthesize_tts(text_trans, lang=tgt_lang, out_path=str(audio_path))
        st.audio(str(audio_path), format="audio/mp3")

        # Persist history (SQLite + CSV)
        save_history_sqlite("upload", "en", text_en, text_trans, tgt_lang)
        save_history_csv("upload", "en", text_en, text_trans, tgt_lang, str(audio_path))

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


def show_history():
    conn = sqlite3.connect("static/history.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM history")
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        st.write(row)

show_history()