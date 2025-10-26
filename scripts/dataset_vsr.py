import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import mediapipe as mp
from vocab import text_to_int, build_vocab

from roi_utils import (
    mouth_center_square as canonical_mouth_center_square,
    _face_bbox_from_landmarks,
    add_roi_padding,
)

# --- Priority 2 --- Standardize these!
MIN_T = 75   # minimum frames after preprocessing
MAX_T = 90   # cap to keep memory bounded

def fit_time_axis(frames, min_t=MIN_T, max_t=MAX_T, img_size=(96, 96)):
    """
    Normalize temporal dimension of frame sequence.
    Args:
        frames: list of 2D arrays (H, W) in temporal order, dtype float32 in [0,1]
        min_t: Minimum number of frames (pad/repeat if shorter)
        max_t: Maximum number of frames (downsample if longer)
        img_size: Expected spatial dimensions (H, W)
    Returns:
        ndarray [T, H, W] with T in [min_t, max_t]
    """
    if len(frames) == 0:
        return np.zeros((min_t, img_size[0], img_size[1]), dtype=np.float32)
    arr = np.stack(frames, axis=0)
    T0 = arr.shape[0]
    if T0 < min_t:
        reps = int(np.ceil(min_t / T0))
        arr = np.repeat(arr, reps, axis=0)[:min_t]
    elif T0 > max_t:
        idx = np.linspace(0, T0 - 1, num=max_t).round().astype(int)
        arr = arr[idx]
    return arr

class MouthROIVideoDataset(Dataset):
    """
    PyTorch Dataset for Visual Speech Recognition (VSR).
    Loads videos, extracts mouth ROI sequences, and prepares for CTC training.
    Uses MediaPipe for face landmark detection and canonical ROI extraction.
    """

    def __init__(self, manifest_csv, roi_size=(96, 96), pad=8,
                 max_frames=150, char2idx=None):
        self.items = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                self.items.append((row["video_path"], row["text"]))
        self.roi_size = roi_size
        self.pad = pad
        self.max_frames = max_frames
        self.char2idx = char2idx or build_vocab()[0]
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.lip_idx = list(range(61, 89))  # NOTE: corrected to 89 for full lips

    def mouth_center_square(self, lms, w, h, scale=1.2):
        return canonical_mouth_center_square(
            lms, w, h, scale=scale, lip_indices=self.lip_idx
        )

    def mouth_bbox(self, lms, w, h):
        x1, y1, x2, y2 = self.mouth_center_square(lms, w, h, scale=1.2)
        x1, y1, x2, y2 = add_roi_padding(x1, y1, x2, y2, w, h, pad_rel=0.03)
        return (x1, y1, x2, y2)

    def roi_seq(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((MIN_T, self.roi_size[0], self.roi_size[1]), dtype=np.float32)
        rois = []
        last_bbox = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0]
                x1, y1, x2, y2 = self.mouth_bbox(lms, w, h)
                last_bbox = (x1, y1, x2, y2)
            elif last_bbox:
                x1, y1, x2, y2 = last_bbox
            else:
                cx, cy = w // 2, h // 2
                sz = min(w, h) // 4
                x1, y1 = cx - sz, cy - sz // 2
                x2, y2 = cx + sz, cy + sz // 2
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.roi_size, interpolation=cv2.INTER_AREA)
            rois.append(resized.astype(np.float32) / 255.0)
            if len(rois) >= self.max_frames:
                break
        cap.release()
        return fit_time_axis(rois, MIN_T, MAX_T, self.roi_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        video_path, text = self.items[i]
        x = self.roi_seq(video_path)
        x = np.expand_dims(x, 1)
        y = text_to_int(text.lower(), self.char2idx)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def pad_collate(batch):
    xs, ys = zip(*batch)
    B = len(xs)
    T_max = max(x.shape[0] for x in xs)
    H, W = xs[0].shape[-2], xs[0].shape[-1]
    xb = torch.zeros((B, T_max, 1, H, W), dtype=torch.float32)
    xlen = torch.zeros(B, dtype=torch.long)
    ycat = []
    ylen = torch.zeros(B, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        t = x.shape[0]
        xb[i, :t] = x
        xlen[i] = t
        y = torch.as_tensor(y, dtype=torch.long)
        ycat.append(y)
        ylen[i] = y.numel()
    yb = torch.cat(ycat, dim=0) if ycat else torch.zeros(0, dtype=torch.long)
    return xb, xlen, yb, ylen
