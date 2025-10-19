# scripts/dataset_vsr.py
import csv, cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from pathlib import Path
from vocab import text_to_int, build_vocab

class MouthROIVideoDataset(Dataset):
    def __init__(self, manifest_csv, roi_size=(96,96), pad=8, max_frames=150, char2idx=None):
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
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.lip_idx = list(range(61,88))

    def _mouth_bbox(self, lms, w, h):
        xs, ys = [], []
        for i in self.lip_idx:
            lm = lms.landmark[i]
            xs.append(int(lm.x * w)); ys.append(int(lm.y * h))
        x1, y1 = max(min(xs)-self.pad, 0), max(min(ys)-self.pad, 0)
        x2, y2 = min(max(xs)+self.pad, w), min(max(ys)+self.pad, h)
        return x1,y1,x2,y2

    def _roi_seq(self, video_path):
        cap = cv2.VideoCapture(video_path)
        rois = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lms = res.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            x1,y1,x2,y2 = self._mouth_bbox(lms, w, h)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, self.roi_size, interpolation=cv2.INTER_AREA)
            rois.append(resized.astype(np.float32)/255.0)
            if len(rois) >= self.max_frames:
                break
        cap.release()
        return np.stack(rois, axis=0) if rois else np.zeros((1, *self.roi_size), np.float32)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        vp, text = self.items[i]
        x = self._roi_seq(vp)            # (T,H,W)
        x = np.expand_dims(x, 1)         # (T,1,H,W)
        y = text_to_int(text.lower(), self.char2idx)  # list[int]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

def pad_collate(batch):
    xs, ys = zip(*batch)
    T = max(x.shape[0] for x in xs)
    H, W = xs[0].shape[-2], xs[0].shape[-1]
    xb = torch.zeros(len(xs), T, 1, H, W, dtype=torch.float32)
    xlen = torch.zeros(len(xs), dtype=torch.long)
    ycat = []
    ylen = torch.zeros(len(xs), dtype=torch.long)
    for i, (x,y) in enumerate(batch):
        t = x.shape[0]
        xb[i, :t] = x
        xlen[i] = t
        ycat.extend(y.tolist())
        ylen[i] = len(y)
    yb = torch.tensor(ycat, dtype=torch.long)
    return xb, xlen, yb, ylen
