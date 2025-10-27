# scripts/dataset_vsr.py
import csv, cv2, numpy as np, torch
from torch.utils.data import Dataset
import mediapipe as mp
from scripts.vocab import text_to_int, build_vocab

# Control the time dimension so CTC has enough steps to spell words.
MIN_T = 64     # minimum frames after preprocessing
MAX_T = 96     # cap to keep memory bounded


def _fit_time_axis(frames, min_t=MIN_T, max_t=MAX_T, img_size=(96, 96)):
    """
    frames: list of 2D arrays (H, W) in temporal order, dtype float32 in [0,1].
    Returns ndarray [T, H, W] with T in [min_t, max_t].
    """
    if len(frames) == 0:
        return np.zeros((min_t, img_size[0], img_size[1]), dtype=np.float32)

    arr = np.stack(frames, axis=0)  # [T0, H, W]
    T0 = arr.shape[0]

    if T0 < min_t:
        reps = int(np.ceil(min_t / T0))
        arr = np.repeat(arr, reps, axis=0)[:min_t]
    elif T0 > max_t:
        idx = np.linspace(0, T0 - 1, num=max_t).round().astype(int)
        arr = arr[idx]
    return arr


class MouthROIVideoDataset(Dataset):
    def __init__(self, manifest_csv, roi_size=(96, 96), pad=8, max_frames=150, char2idx=None):
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
        # MediaPipe lip landmarks
        self.lip_idx = list(range(61, 88))

    # ---- New helpers replacing original _mouth_bbox logic ----
    def _face_bbox_from_landmarks(self, lms, w, h, pad_rel=0.08):
        xs = [int(lm.x * w) for lm in lms.landmark]
        ys = [int(lm.y * h) for lm in lms.landmark]
        x1, y1 = max(min(xs), 0), max(min(ys), 0)
        x2, y2 = min(max(xs), w), min(max(ys), h)
        padx = int((x2 - x1) * pad_rel)
        pady = int((y2 - y1) * pad_rel)
        return [max(0, x1 - padx), max(0, y1 - pady), min(w, x2 + padx), min(h, y2 + pady)]

    def _mouth_center_square(self, lms, w, h, scale=1.0):
        lip_idx = list(range(61, 88))
        xs = [(lms.landmark[i].x * w) for i in lip_idx]
        ys = [(lms.landmark[i].y * h) for i in lip_idx]
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
        size = int(max(lip_w * 2.6, face_w * 0.28))
        size = int(size * float(scale))
        half = max(8, size // 2)
        x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return [x1, y1, x2, y2]

    def _mouth_bbox(self, lms, w, h):
        # upstream compatibility: return tight bbox used in training
        x1, y1, x2, y2 = self._mouth_center_square(lms, w, h, scale=1.0)
        pad_px = max(2, int(min(w, h) * 0.03))
        x1, y1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
        x2, y2 = min(w, x2 + pad_px), min(h, y2 + pad_px)
        return x1, y1, x2, y2
    # ----------------------------------------------------------

    def _roi_seq(self, video_path):
        cap = cv2.VideoCapture(video_path)
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
                x1, y1, x2, y2 = self._mouth_bbox(lms, w, h)
                last_bbox = (x1, y1, x2, y2)
            elif last_bbox:
                x1, y1, x2, y2 = last_bbox
            else:
                # Fallback: center crop if no face for first frames
                cx, cy = w // 2, h // 2
                sz = min(w, h) // 4
                x1, y1, x2, y2 = cx - sz, cy - sz // 2, cx + sz, cy + sz // 2

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

        # Guarantee a usable time dimension for CTC
        return _fit_time_axis(rois, MIN_T, MAX_T, self.roi_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vp, text = self.items[i]
        x = self._roi_seq(vp)              # [T, H, W], float32 in [0,1]
        x = np.expand_dims(x, 1)           # [T, 1, H, W]
        y = text_to_int(text.lower(), self.char2idx)  # list[int], no blank in labels
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def pad_collate(batch):
    # Batch is list of (x[T,1,H,W], y[L])
    xs, ys = zip(*batch)
    B = len(xs)
    T_max = max(x.shape[0] for x in xs)
    H, W = xs[0].shape[-2], xs[0].shape[-1]

    xb = torch.zeros((B, T_max, 1, H, W), dtype=torch.float32)
    xlen = torch.zeros(B, dtype=torch.long)

    # Flatten targets for CTCLoss and keep lengths
    ycat = []
    ylen = torch.zeros(B, dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        t = x.shape[0]
        xb[i, :t] = x
        xlen[i] = t
        y = torch.as_tensor(y, dtype=torch.long)
        ycat.append(y)
        ylen[i] = y.numel()

    yb = torch.cat(ycat, dim=0) if len(ycat) > 0 else torch.zeros(0, dtype=torch.long)
    return xb, xlen, yb, ylen
