
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp

st.subheader("Webcam (landmarks)")

# Toggle audio off for this demo
MEDIA_CONSTRAINTS = {"video": True, "audio": False}

class LandmarkProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if res.multi_face_landmarks:
            for face_landmarks in res.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style(),
                )
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="webcam-lips",
    video_processor_factory=LandmarkProcessor,
    media_stream_constraints=MEDIA_CONSTRAINTS,
)

