# app.py - FULL ENHANCED VERSION (preserves your features; adds Presentation Mode, IST everywhere needed,
# confidence chips + char-heat, TTS guardrail, and Download debug bundle zip)
# Run: streamlit run app.py

import os, uuid, pathlib, time, csv, json, zipfile, io
import sqlite3
import torch
import streamlit as st
import av, cv2
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import mediapipe as mp

from models import LipReaderModel

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="SilentSpeaker", page_icon="🎙️", layout="centered", initial_sidebar_state="collapsed")
st.title("🎙️ SilentSpeaker")
st.markdown("##### Read lips. Generate text. Give voice to silence.")

# -----------------------------
# Paths and folders
# -----------------------------
BASE = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
AUDIO_DIR = STATIC_DIR / "audio"
EXPORT_DIR = STATIC_DIR / "exports"  # NEW: for debug bundles
DB_PATH = STATIC_DIR / "history.db"
CSV_PATH = STATIC_DIR / "history.csv"

for p in [STATIC_DIR, UPLOAD_DIR, AUDIO_DIR, EXPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Translation helper
# -----------------------------
def translate_text(text: str, target_lang: str = "hi") -> str:
    try:
        from googletrans import Translator
        tr = Translator()
        return tr.translate(text or " ", dest=target_lang).text
    except Exception:
        return text or " "

# -----------------------------
# Database helpers
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
    # UPDATED: show IST for latest row
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT id, datetime(ts + 19800, 'unixepoch') AS time_ist, source, text_en, tgt_lang FROM history ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        con.close()
        if row:
            st.info(f"Latest row → id={row[0]}, time(IST)={row[1]}, source={row[2]}, tgt={row[4]}")
    except Exception as e:
        st.warning(f"Could not read latest row: {e}")

init_db()
migrate_history_schema()

# -----------------------------
# Cache model
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    ckpt = str(BASE / "models" / "checkpoints" / "lip_reader.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return LipReaderModel(checkpoint_path=ckpt, device=dev)
    except TypeError:
        return LipReaderModel()

model = get_model()

# -----------------------------
# Confidence chips and char visualization (UI helpers)
# -----------------------------
def display_confidence_detailed(conf: float, text: str = "", char_confidences: list = None):
    # Overall confidence bar
    if conf >= 0.5:
        st.success(f"🎯 **High Confidence:** {conf:.1%}")
        st.caption("✅ Prediction is reliable")
    elif conf >= 0.2:
        st.warning(f"⚠️ **Medium Confidence:** {conf:.1%}")
        st.caption("⚠️ Prediction may have errors - verify results")
    else:
        st.error(f"❌ **Low Confidence:** {conf:.1%}")
        st.caption("❌ Prediction is unreliable - poor video quality or unclear speech")
    # Character-level confidence visualization
    if char_confidences and len(char_confidences) > 0:
        with st.expander("🔍 Character-Level Confidence", expanded=False):
            st.caption("Characters highlighted by confidence: 🟢 High (>0.7) | 🟡 Medium (0.4-0.7) | 🔴 Low (<0.4)")
            html_chars = []
            for char, conf_val in char_confidences:
                if conf_val >= 0.7:
                    color_code = "#28a745"
                elif conf_val >= 0.4:
                    color_code = "#ffc107"
                else:
                    color_code = "#dc3545"
                display_char = "&nbsp;" if char == " " else char
                html_chars.append(
                    f'<span style="color: {color_code}; font-weight: bold; font-size: 1.2em;" '
                    f'title="{char} ({conf_val:.2%})">{display_char}</span>'
                )
            st.markdown(" ".join(html_chars), unsafe_allow_html=True)
            st.markdown("##### Confidence Distribution")
            low = sum(1 for _, c in char_confidences if c < 0.4)
            med = sum(1 for _, c in char_confidences if 0.4 <= c < 0.7)
            high = sum(1 for _, c in char_confidences if c >= 0.7)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Low", f"{low} chars", f"{(low/max(1,len(char_confidences)))*100:.0f}%")
            with col2:
                st.metric("🟡 Medium", f"{med} chars", f"{(med/max(1,len(char_confidences)))*100:.0f}%")
            with col3:
                st.metric("🟢 High", f"{high} chars", f"{(high/max(1,len(char_confidences)))*100:.0f}%")

# -----------------------------
# Debug bundle exporter (annotated.mp4 + bbox.csv + debug.json) as a zip
# -----------------------------
def make_debug_bundle(video_stem: str, annotated_frames, bboxes, debug_info):
    export_dir = EXPORT_DIR / video_stem
    export_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = export_dir / f"{video_stem}_annotated.mp4"
    if annotated_frames:
        h, w = annotated_frames[0].shape[:2]
        out = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        for fr in annotated_frames:
            out.write(fr)
        out.release()
    bbox_path = export_dir / f"{video_stem}_bbox.csv"
    with open(bbox_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame","x1","y1","x2","y2"])
        for i, (x1,y1,x2,y2) in enumerate(bboxes):
            wcsv.writerow([i, x1, y1, x2, y2])
    debug_path = export_dir / f"{video_stem}_debug.json"
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_info or {}, f, indent=2)
    # Build in-memory zip for download button
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if annotated_frames:
            z.write(annotated_path, arcname=annotated_path.name)
        z.write(bbox_path, arcname=bbox_path.name)
        z.write(debug_path, arcname=debug_path.name)
    buf.seek(0)
    return annotated_path, bbox_path, debug_path, buf

# -----------------------------
# Sidebar with enhanced model status and Presentation toggle
# -----------------------------
with st.sidebar:
    st.caption("Database")
    st.code(str(DB_PATH))
    try:
        ts = os.path.getmtime(DB_PATH)
        st.caption(f"Last modified: {time.ctime(ts)}")
    except Exception:
        st.caption("DB not found")

    st.subheader("Mode")
    presentation_mode = st.toggle("🎤 Presentation Mode", value=False, help="Hide debug panels and enlarge key elements")

    st.caption("Model Status")
    ckpt_path_default = BASE / "models" / "checkpoints" / "lip_reader.pt"
    if model.demo_mode:
        st.error("⚠️ MODEL IN DEMO MODE")
        if hasattr(model, 'load_error') and model.load_error:
            st.warning(f"Error: {model.load_error}")
        st.caption("The model will return placeholder text only")
    else:
        st.success("✅ Model loaded successfully")

    st.code(f"Checkpoint: {ckpt_path_default.name}")
    st.code(f"Exists: {ckpt_path_default.exists()}")
    st.code(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    st.code(f"Demo mode: {model.demo_mode}")

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

# ========================================
# UPLOAD TAB (preserved + enhanced)
# ========================================
with tab_upload:
    st.markdown("### 📤 Upload Video for Lip Reading")
    uploaded_video = st.file_uploader("Upload a video of a person speaking", type=["mp4","mov","avi","mpg","mkv"])

    # Keep your two-column layout; add a third toggle for Large text
    col_lang, col_debug, col_present = st.columns([2, 1, 1])
    with col_lang:
        tgt_lang = st.selectbox("TTS language", ["hi","en","bn","ta","te"], index=0)
    with col_debug:
        # Default Debug off in Presentation Mode for a clean flow
        enable_debug = st.checkbox("🔧 Enable Debug Mode", value=(not presentation_mode))
    with col_present:
        large_text = st.checkbox("Large text", value=presentation_mode)

    if uploaded_video is not None:
        video_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(str(video_path))

        with st.spinner("🔍 Predicting text from lips..."):
            if enable_debug:
                result = model.predict_with_debug(str(video_path))
            else:
                result = model.predict(str(video_path))
                # Ensure keys exist to avoid KeyErrors in UI paths
                result.setdefault("roi_frames", [])
                result.setdefault("annotated_frames", [])
                result.setdefault("debug_info", {})
                result.setdefault("logits_sample", [])
                result.setdefault("char_confidences", [])
                result.setdefault("bboxes", [])

        text_en = result.get("text", "")
        conf = float(result.get("conf", 0.0))
        roi_frames = result.get("roi_frames", [])
        annotated_frames = result.get("annotated_frames", [])
        debug_info = result.get("debug_info", {})
        logits_sample = result.get("logits_sample", [])
        char_confidences = result.get("char_confidences", [])
        bboxes = result.get("bboxes", [])

        # Optional larger typography for presentation
        if large_text:
            st.markdown("<style>.bigtext{font-size:1.35rem;}</style>", unsafe_allow_html=True)

        st.success("✅ Prediction Complete!")
        display_confidence_detailed(conf, text_en, char_confidences)

        st.subheader("📝 Predicted Text (English)")
        st.markdown(f"<div class='bigtext'>{text_en or '[No text detected]'}</div>", unsafe_allow_html=True)

        # Debug section (preserved and expanded; auto-hidden when Presentation Mode unless user enables it)
        if enable_debug and (roi_frames or debug_info):
            with st.expander("🔧 **Debug Information**", expanded=True if not presentation_mode else False):

                st.markdown("#### 📊 Processing Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", debug_info.get("total_frames", 0))
                with col2:
                    st.metric("Detected Faces", debug_info.get("detected_frames", 0))
                with col3:
                    st.metric("ROI Extracted", debug_info.get("roi_count", 0))
                with col4:
                    st.metric("Detection Rate", debug_info.get("detection_rate", "N/A"))

                # ROI Preview (B&W 96x96)
                if roi_frames:
                    st.markdown("---")
                    st.markdown("#### 🔍 ROI Frames Preview (Grayscale 96×96)")
                    st.caption("These are the preprocessed mouth regions fed to the AI model")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(roi_frames[0], caption=f"First (1/{len(roi_frames)})", clamp=True, width=150)
                    with col2:
                        mid_idx = len(roi_frames) // 2
                        st.image(roi_frames[mid_idx], caption=f"Middle ({mid_idx}/{len(roi_frames)})", clamp=True, width=150)
                    with col3:
                        st.image(roi_frames[-1], caption=f"Last ({len(roi_frames)}/{len(roi_frames)})", clamp=True, width=150)

                    show_all_roi = st.checkbox("📋 Show all ROI frames (may be slow for long videos)")
                    if show_all_roi:
                        st.markdown("##### All ROI Frames")
                        st.caption(f"Showing {len(roi_frames)} frames")
                        cols = st.columns(6)
                        for i, roi in enumerate(roi_frames[:60]):
                            with cols[i % 6]:
                                st.image(roi, caption=f"#{i+1}", width=100, clamp=True)
                        if len(roi_frames) > 60:
                            st.caption(f"⚠️ Showing first 60 of {len(roi_frames)} frames")

                # Annotated video frames (COLOR with green boxes)
                if annotated_frames:
                    st.markdown("---")
                    st.markdown("#### 📹 Bounding Box Visualization (Color)")
                    st.caption("Green box shows detected mouth region on original frames")

                    # Save annotated video (for immediate player)
                    annotated_video_path = UPLOAD_DIR / f"{video_path.stem}_annotated.mp4"
                    h, w = annotated_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(str(annotated_video_path), fourcc, 20.0, (w, h))
                    for frame in annotated_frames:
                        out.write(frame)
                    out.release()

                    st.video(str(annotated_video_path))

                    st.markdown("##### Sample Annotated Frames (Color)")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(cv2.cvtColor(annotated_frames[0], cv2.COLOR_BGR2RGB), caption="First", use_column_width=True)
                    with col2:
                        mid = len(annotated_frames) // 2
                        st.image(cv2.cvtColor(annotated_frames[mid], cv2.COLOR_BGR2RGB), caption="Middle", use_column_width=True)
                    with col3:
                        st.image(cv2.cvtColor(annotated_frames[-1], cv2.COLOR_BGR2RGB), caption="Last", use_column_width=True)

                    show_all_annotated = st.checkbox("📋 Show all annotated frames")
                    if show_all_annotated:
                        st.markdown("##### All Annotated Frames")
                        st.caption(f"Showing {len(annotated_frames)} frames")
                        cols = st.columns(4)
                        for i, frame in enumerate(annotated_frames[:40]):
                            with cols[i % 4]:
                                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"#{i+1}", use_column_width=True)
                        if len(annotated_frames) > 40:
                            st.caption(f"⚠️ Showing first 40 of {len(annotated_frames)} frames")

                    # NEW: Download debug bundle (zip)
                    try:
                        stem = video_path.stem
                        _, _, _, zip_buf = make_debug_bundle(stem, annotated_frames, bboxes, debug_info)
                        st.download_button(
                            "⬇️ Download debug bundle (annotated.mp4 + bbox.csv + debug.json)",
                            data=zip_buf,
                            file_name=f"{stem}_debug_bundle.zip",
                            mime="application/zip"
                        )
                    except Exception as e:
                        st.warning(f"Could not prepare debug bundle: {e}")
                else:
                    st.warning("⚠️ No annotated frames available. Check face detection.")

                # Model internals
                st.markdown("---")
                st.markdown("#### 🧠 Model Internals")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Time axis:** {debug_info.get('time_axis_after_fit', 'N/A')}")
                    st.write(f"**Input shape:** {debug_info.get('model_input_shape', 'N/A')}")
                with col2:
                    st.write(f"**Logits shape:** {debug_info.get('logits_shape', 'N/A')}")
                    st.write(f"**Vocab size:** {len(getattr(model, 'idx2char', [])) or 'N/A'}")

        # Translation & TTS with guardrail (do not remove your existing path; gate it by confidence)
        if conf < 0.2:
            st.info("TTS disabled due to low confidence (< 20%). Improve input or re-upload to enable speech.")
            text_out = ""
            audio_path = ""
        else:
            with st.spinner("🌐 Translating..."):
                text_out = translate_text(text_en, target_lang=tgt_lang)
            st.subheader("🌐 Translated Text")
            st.markdown(f"<div class='bigtext'>{text_out or ''}</div>", unsafe_allow_html=True)
            with st.spinner("🔊 Generating speech..."):
                audio_path = AUDIO_DIR / f"{uuid.uuid4().hex}.mp3"
                try:
                    gTTS(text_out or " ", lang=tgt_lang).save(str(audio_path))
                    st.audio(str(audio_path), format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS failed: {e}")
                    audio_path = ""

        # Save to history (preserved)
        if save_history_row_safe("upload", "en", text_en, text_out, tgt_lang, str(audio_path), str(video_path)):
            show_latest_row()

# ========================================
# WEBCAM TAB - ENHANCED (preserved)
# ========================================
with tab_webcam:
    st.markdown("### 📹 Live Webcam Lip Reading")
    st.caption("Real-time face detection with mouth ROI tracking")

    col1, col2 = st.columns([2, 1])
    with col1:
        tgt_lang_cam = st.selectbox("TTS language", ["hi","en","bn","ta","te"], index=0, key="tgt_cam")
    with col2:
        show_fps = st.checkbox("📊 Show FPS", value=True)

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
            self.last_roi = None
            self.prev_bbox = None
            self.smooth_alpha = 0.6
            self.lip_idx = list(range(61, 88))
            self.frame_count = 0
            self.detection_count = 0
            self.last_time = time.time()
            self.fps = 0.0

        def _face_bbox_from_landmarks(self, lms, w, h, pad_rel=0.08):
            xs = [int(lm.x * w) for lm in lms.landmark]
            ys = [int(lm.y * h) for lm in lms.landmark]
            x1, y1 = max(min(xs), 0), max(min(ys), 0)
            x2, y2 = min(max(xs), w), min(max(ys), h)
            padx = int((x2 - x1) * pad_rel)
            pady = int((y2 - y1) * pad_rel)
            return [max(0, x1 - padx), max(0, y1 - pady), min(w, x2 + padx), min(h, y2 + pady)]

        def _mouth_center_square(self, lms, w, h, scale=1.0):
            xs = [(lms.landmark[i].x * w) for i in self.lip_idx]
            ys = [(lms.landmark[i].y * h) for i in self.lip_idx]
            if len(xs) == 0:
                fb = self._face_bbox_from_landmarks(lms, w, h)
                fw = fb[2] - fb[0]
                cx = fb[0] + fw // 2
                cy = fb[1] + (fb[3] - fb[1]) // 2
                size = int(fw * 0.35)
                half = size // 2
                return [max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)]
            cx = int(sum(xs) / len(xs)); cy = int(sum(ys) / len(ys))
            lip_w = max(1.0, (max(xs) - min(xs)))
            face_bbox = self._face_bbox_from_landmarks(lms, w, h)
            face_w = float(face_bbox[2] - face_bbox[0])
            size = int(max(lip_w * 2.8, face_w * 0.38))
            size = int(size * float(scale))
            half = max(8, size // 2)
            x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return [x1, y1, x2, y2]

        def _smooth_bbox(self, bbox):
            if self.prev_bbox is None:
                self.prev_bbox = bbox
                return bbox
            a = float(self.smooth_alpha)
            sm = [int(round(a * bbox[i] + (1 - a) * self.prev_bbox[i])) for i in range(4)]
            self.prev_bbox = sm
            return sm

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            if res.multi_face_landmarks:
                self.detection_count += 1
                for face_landmarks in res.multi_face_landmarks:
                    self.mp_draw.draw_landmarks(
                        img, face_landmarks,
                        self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style()
                    )
                    x1, y1, x2, y2 = self._mouth_center_square(face_landmarks, img.shape[1], img.shape[0], scale=1.0)
                    pad_px = max(2, int(min(img.shape[0], img.shape[1]) * 0.03))
                    x1, y1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
                    x2, y2 = min(img.shape[1], x2 + pad_px), min(img.shape[0], y2 + pad_px)
                    x1, y1, x2, y2 = self._smooth_bbox([x1, y1, x2, y2])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"ROI: {x2-x1}x{y2-y1}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    crop = img[y1:y2, x1:x2]
                    if crop.size != 0:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
                        try:
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            resized = clahe.apply(resized.astype('uint8'))
                        except Exception:
                            resized = resized.astype('uint8')
                        self.last_roi = resized
            if show_fps:
                cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            self.buffer.append(img)
            if len(self.buffer) > 120:
                self.buffer = self.buffer[-120:]
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        def get_buffer_and_clear(self):
            frames = self.buffer[:]
            self.buffer = []
            return frames

        def get_last_roi(self):
            return self.last_roi

        def get_debug_info(self):
            return {
                "buffer_size": len(self.buffer),
                "last_roi_available": self.last_roi is not None,
                "prev_bbox": self.prev_bbox,
                "detection_count": self.detection_count,
                "fps": self.fps
            }

    RTC_CFG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:global.stun.twilio.com:3478"]},
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

    # Live debug info (preserved)
    if ctx and ctx.video_processor:
        st.markdown("---")
        st.markdown("### 📊 Live Statistics")
        col1, col2, col3 = st.columns(3)
        debug_info = ctx.video_processor.get_debug_info()
        with col1:
            last_roi = ctx.video_processor.get_last_roi()
            if last_roi is not None:
                st.image(last_roi, caption="🔍 Live Mouth ROI (96×96)", width=200)
            else:
                st.info("👤 Waiting for face detection...")
        with col2:
            st.metric("Buffer Frames", debug_info["buffer_size"], help="Frames stored for processing")
            st.metric("FPS", f"{debug_info['fps']:.1f}", help="Current frame rate")
        with col3:
            st.metric("ROI Status", "✅ Detected" if debug_info["last_roi_available"] else "⏳ Searching")
            st.metric("Total Detections", debug_info["detection_count"])

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
        char_confs = res.get("char_confidences", [])
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
        return {
            "text": text_en,
            "conf": conf,
            "char_confidences": char_confs,
            "text_out": text_out,
            "audio": str(audio_path),
            "video": str(temp_path)
        }

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Transcript", use_container_width=True):
            if ctx and ctx.video_processor:
                frames = ctx.video_processor.get_buffer_and_clear()
                with st.spinner("Processing..."):
                    result = flush_and_process(frames)
                if result:
                    auto_result_placeholder.success("✅ Transcript saved!")
                    st.subheader("📝 Predicted Text")
                    st.write(result["text"])
                    display_confidence_detailed(result["conf"], result["text"], result["char_confidences"])
                    st.subheader("🌐 Translated")
                    st.write(result["text_out"])
                    if result["audio"]:
                        st.audio(result["audio"], format="audio/mp3")
            else:
                st.warning("No frames captured. Start the stream first.")

    with col2:
        if st.button("🎬 Capture Single Frame", use_container_width=True):
            if ctx and ctx.video_processor:
                last_roi = ctx.video_processor.get_last_roi()
                if last_roi is not None:
                    st.image(last_roi, caption="Captured ROI Frame", width=200)
                    st.success("✅ Frame captured!")
                else:
                    st.warning("No face detected in current frame")

# ========================================
# HISTORY TAB (kept; IST already applied)
# ========================================
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
                "SELECT id, datetime(ts + 19800, 'unixepoch') AS time, source, text_en, text_trans, tgt_lang "
                "FROM history ORDER BY id DESC LIMIT 20", con)
        except Exception:
            cur = con.cursor()
            cur.execute("SELECT id, ts, source, text_en, text_trans, tgt_lang FROM history ORDER BY id DESC LIMIT 20")
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

st.caption("💡 Tip: Presentation Mode hides debug panels and enlarges key elements; toggle Debug Mode anytime to inspect internals.")
