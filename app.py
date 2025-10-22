# app.py
# Run: streamlit run app.py

import os, uuid, pathlib, time, csv
import sqlite3
import torch
import streamlit as st
import av, cv2
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import mediapipe as mp

# Your trained model wrapper must exist in models.py and return {"text": str, "conf": float}
from models import LipReaderModel

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="SilentSpeaker", page_icon="🎙️", layout="centered")
st.title("🎙️ SilentSpeaker")
st.markdown("##### Read lips. Generate text. Give voice to silence.")

# -----------------------------
# Paths and folders
# -----------------------------
BASE = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
AUDIO_DIR = STATIC_DIR / "audio"
DB_PATH = STATIC_DIR / "history.db"
CSV_PATH = STATIC_DIR / "history.csv"

for p in [STATIC_DIR, UPLOAD_DIR, AUDIO_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Lightweight translation helper
# -----------------------------
def translate_text(text: str, target_lang: str = "hi") -> str:
    try:
        from googletrans import Translator
        tr = Translator()
        return tr.translate(text or " ", dest=target_lang).text
    except Exception:
        return text or " "

# -----------------------------
# SQLite + CSV history helpers
# -----------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER,
        source TEXT,
        src_lang TEXT,
        text_en TEXT,
        text_trans TEXT,
        tgt_lang TEXT,
        audio_path TEXT,
        video_path TEXT
      )
    """)
    con.commit()
    con.close()

def migrate_history_schema():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("PRAGMA table_info(history)")
    cols = [r[1] for r in cur.fetchall()]
    if "audio_path" not in cols:
        cur.execute("ALTER TABLE history ADD COLUMN audio_path TEXT")
    if "video_path" not in cols:
        cur.execute("ALTER TABLE history ADD COLUMN video_path TEXT")
    con.commit()
    con.close()

def save_history_row(source, src_lang, text_en, text_trans, tgt_lang, audio_path, video_path):
    ts = int(time.time())
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO history(ts, source, src_lang, text_en, text_trans, tgt_lang, audio_path, video_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, source, src_lang, text_en, text_trans, tgt_lang, audio_path, video_path),
    )
    con.commit()
    con.close()
    # CSV backup
    new_file = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts","source","src_lang","text_en","text_trans","tgt_lang","audio_path","video_path"])
        w.writerow([ts, source, src_lang, text_en, text_trans, tgt_lang, audio_path, video_path])

def save_history_row_safe(source, src_lang, text_en, text_trans, tgt_lang, audio_path, video_path):
    try:
        save_history_row(source, src_lang, text_en, text_trans, tgt_lang, audio_path, video_path)
        st.toast("Saved to history ✅", icon="✅")
        return True
    except Exception as e:
        st.toast(f"History save failed: {e}", icon="⚠️")
        return False

def show_latest_row():
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT id, ts, source, text_en, tgt_lang FROM history ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        con.close()
        if row:
            st.info(f"Latest row → id={row[0]}, source={row[2]}, tgt={row[4]}")
    except Exception as e:
        st.warning(f"Could not read latest row: {e}")

init_db()
migrate_history_schema()

# -----------------------------
# Cache model (load your checkpoint here)
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    # Change this to an absolute path if you store the file elsewhere
    ckpt = str(BASE / "models" / "checkpoints" / "lip_reader.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Prefer passing checkpoint and device; fall back to default constructor if wrapper signature differs
    try:
        return LipReaderModel(checkpoint_path=ckpt, device=dev)
    except TypeError:
        return LipReaderModel()

model = get_model()

# -----------------------------
# Sidebar: DB, model, and controls
# -----------------------------
with st.sidebar:
    st.caption("Database")
    st.code(str(DB_PATH))
    try:
        ts = os.path.getmtime(DB_PATH)
        st.caption(f"Last modified: {time.ctime(ts)}")
    except Exception:
        st.caption("DB not found")

    st.caption("Model status")
    ckpt_path_default = BASE / "models" / "checkpoints" / "lip_reader.pt"
    st.code(f"Checkpoint: {ckpt_path_default.name} → {'found' if ckpt_path_default.exists() else 'missing'}")
    st.caption(f"Device → {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Reload model"):
            get_model.clear()
            st.cache_resource.clear()
            st.rerun()
    with colB:
        if st.button("Reset history DB"):
            try:
                if DB_PATH.exists():
                    DB_PATH.unlink()
                init_db()
                migrate_history_schema()
                st.success("History DB reset.")
            except Exception as e:
                st.error(f"Reset failed: {e}")

# -----------------------------
# Tabs
# -----------------------------
tab_upload, tab_webcam, tab_history = st.tabs(["Upload", "Webcam", "History"])

# -----------------------------
# Upload tab
# -----------------------------
with tab_upload:
    uploaded_video = st.file_uploader("Upload a video of a person speaking", type=["mp4","mov","avi","mpg","mkv"])
    tgt_lang = st.selectbox("TTS language", ["hi","en","bn","ta","te"], index=0)

    if uploaded_video is not None:
        # Persist upload
        video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(str(video_path))

        with st.spinner("🔍 Predicting text from lips..."):
            result = model.predict(str(video_path))  # expects {"text": str, "conf": float}
        text_en = result.get("text", "")
        conf = float(result.get("conf", 0.0))

        st.success("✅ Prediction Complete!")
        st.subheader("📝 Predicted Text")
        st.write(text_en or "")
        st.caption(f"Confidence: {conf:.3f}")

        with st.spinner("🌐 Translating..."):
            text_out = translate_text(text_en, target_lang=tgt_lang)
        st.subheader("🌐 Translated")
        st.write(text_out or "")

        with st.spinner("🔊 Generating speech..."):
            audio_path = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
            try:
                gTTS(text_out or " ", lang=tgt_lang).save(str(audio_path))
                st.audio(str(audio_path), format="audio/mp3")
            except Exception as e:
                st.error(f"TTS failed: {e}")
                audio_path = ""

        if save_history_row_safe("upload", "en", text_en, text_out, tgt_lang, str(audio_path), str(video_path)):
            show_latest_row()

# -----------------------------
# Webcam tab (streamlit-webrtc)
# -----------------------------
with tab_webcam:
    st.caption("Live camera with lip/face contours and Save Transcript")
    tgt_lang_cam = st.selectbox("TTS language (webcam)", ["hi","en","bn","ta","te"], index=0, key="tgt_cam")
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
            self.buffer = []

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
            self.buffer.append(img)
            if len(self.buffer) > 120:
                self.buffer = self.buffer[-120:]
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def get_buffer_and_clear(self):
            frames = self.buffer[:]
            self.buffer = []
            return frames

    # Explicit ICE servers for hosted deployments (replace TURN with real credentials or env vars)
    RTC_CFG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"]},
            {
                "urls": ["turn:your.turn.server:3478", "turns:your.turn.server:5349"],
                "username": os.environ.get("TURN_USERNAME", "user"),
                "credential": os.environ.get("TURN_CREDENTIAL", "pass")
            }
        ]
    })

    ctx = webrtc_streamer(
        key="webcam-lips",
        video_processor_factory=LandmarkProcessor,
        media_stream_constraints=MEDIA,
        rtc_configuration=RTC_CFG,
    )

    if "was_playing" not in st.session_state:
        st.session_state.was_playing = False

    auto_result_placeholder = st.empty()

    def flush_and_process(frames):
        if not frames:
            return None
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_webcam.mp4"
        out = cv2.VideoWriter(str(temp_path), fourcc, 20.0, (w, h))
        for f in frames:
            out.write(f)
        out.release()

        res = model.predict(str(temp_path))
        text_en = res.get("text", "")
        conf = float(res.get("conf", 0.0))
        text_out = translate_text(text_en, target_lang=tgt_lang_cam)

        audio_path = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
        try:
            gTTS(text_out or " ", lang=tgt_lang_cam).save(str(audio_path))
        except Exception as e:
            st.error(f"TTS failed: {e}")
            audio_path = ""

        ok = save_history_row_safe("webcam", "en", text_en, text_out, tgt_lang_cam, str(audio_path), str(temp_path))
        if ok:
            show_latest_row()
        return {"text": text_en, "conf": conf, "text_out": text_out, "audio": str(audio_path), "video": str(temp_path)}

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Transcript (Webcam)"):
            if ctx and ctx.video_processor:
                frames = ctx.video_processor.get_buffer_and_clear()
                result = flush_and_process(frames)
                if result:
                    auto_result_placeholder.success("✅ Saved from webcam")
                    st.subheader("📝 Predicted Text (Webcam)")
                    st.write(result["text"])
                    st.caption(f"Confidence: {result['conf']:.3f}")
                    st.subheader("🌐 Translated (Webcam)")
                    st.write(result["text_out"])
                    if result["audio"]:
                        st.audio(result["audio"], format="audio/mp3")
                else:
                    st.warning("No frames captured. Try again.")

    with col2:
        st.write("Stop the stream to auto-save the last few seconds.")
        playing = ctx.state.playing if ctx else False
        if st.session_state.was_playing and not playing:
            if ctx and ctx.video_processor:
                frames = ctx.video_processor.get_buffer_and_clear()
                result = flush_and_process(frames)
                if result:
                    auto_result_placeholder.success("✅ Auto-saved on Stop")
        st.session_state.was_playing = playing

# -----------------------------
# History tab
# -----------------------------
with tab_history:
    st.caption("Saved transcripts are stored here:")
    st.write(f"SQLite DB: {DB_PATH}")
    st.write(f"CSV log: {CSV_PATH}")

    try:
        con = sqlite3.connect(DB_PATH)
        df = None
        try:
            import pandas as pd
            df = pd.read_sql_query(
                "SELECT id, datetime(ts, 'unixepoch') AS time, source, text_en, text_trans, tgt_lang, audio_path, video_path "
                "FROM history ORDER BY id DESC LIMIT 20", con)
        except Exception:
            cur = con.cursor()
            cur.execute("SELECT id, ts, source, text_en, text_trans, tgt_lang, audio_path, video_path FROM history ORDER BY id DESC LIMIT 20")
            rows = cur.fetchall()
            if rows:
                st.write("Recent entries:")
                for r in rows:
                    st.write(str(r))
        finally:
            con.close()
        if df is not None:
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not read history: {e}")

st.caption("Tip: After editing this file, restart Streamlit or use the sidebar Reload model to clear caches.")
