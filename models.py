# models.py (drop-in replacement)
import cv2, numpy as np, math
from pathlib import Path


class LipReaderModel:
    def __init__(self, checkpoint_path=None, roi_size=(96,96), device="cpu", blank_id=0, use_beam=False):
        import mediapipe as mp
        self.mp = mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.roi_size = roi_size
        self.blank_id = blank_id
        self.lip_indices = list(range(61, 88))
        self.prev_bbox = None
        self.smooth_alpha = 0.6
        self.device = device
        self.use_beam = use_beam

        base_dir = Path(__file__).parent.resolve()
        default_ckpt = base_dir / "checkpoints" / "lip_reader.pt"
        self.ckpt_path = Path(checkpoint_path).resolve() if checkpoint_path else default_ckpt

        self.demo_mode = False
        self.model = None
        self.idx2char = list(" " + "abcdefghijklmnopqrstuvwxyz" + "'")
        self.torch = None
        self.F = None
        self.decoder = None

        try:
            import torch, torch.nn.functional as F
            self.torch, self.F = torch, F
            if self.ckpt_path.exists():
                ckpt = torch.load(str(self.ckpt_path), map_location=device)
                if isinstance(ckpt, dict) and "idx2char" in ckpt:
                    self.idx2char = ckpt["idx2char"]
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    from scripts.model_vsr import CNN3D_BiGRU
                    self.model = CNN3D_BiGRU(vocab_size=len(self.idx2char))
                    self.model.load_state_dict(ckpt["state_dict"], strict=False)
                else:
                    self.model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.to(self.device).eval()
                if self.use_beam:
                    try:
                        from torchaudio.models.decoder import CTCDecoder
                        self.decoder = CTCDecoder(
                            lexicon=None, tokens=self.idx2char, lm=None, nbest=1, beam_size=10, blank_token=self.idx2char[0]
                        )
                    except Exception:
                        self.decoder = None
                print(f"[SilentSpeaker] Loaded checkpoint: {self.ckpt_path}")
            else:
                self.demo_mode = True
                print(f"[SilentSpeaker] Checkpoint not found, demo_mode=True: {self.ckpt_path}")
        except Exception as e:
            self.demo_mode = True
            print(f"[SilentSpeaker] Model load error, demo_mode=True: {e}")

    # -------- Revised ROI helpers --------
    def _face_bbox_from_landmarks(self, lms, w, h, pad_rel=0.08):
        xs = [int(lm.x * w) for lm in lms.landmark]
        ys = [int(lm.y * h) for lm in lms.landmark]
        x1, y1 = max(min(xs), 0), max(min(ys), 0)
        x2, y2 = min(max(xs), w), min(max(ys), h)
        padx = int((x2 - x1) * pad_rel)
        pady = int((y2 - y1) * pad_rel)
        return [max(0, x1 - padx), max(0, y1 - pady), min(w, x2 + padx), min(h, y2 + pady)]

    def _mouth_center_square(self, lms, w, h, scale=1.0):
        lip_idx = self.lip_indices
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

        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))

        lip_w = max(1.0, (max(xs) - min(xs)))
        face_bbox = self._face_bbox_from_landmarks(lms, w, h)
        face_w = float(face_bbox[2] - face_bbox[0])

        size = int(max(lip_w * 2.6, face_w * 0.28))
        size = int(size * float(scale))
        half = max(8, size // 2)

        x1, y1 = cx - half, cy - half
        x2, y2 = cx + half, cy + half

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
    # ------------------------------------

    # -------- Time fitting helper --------
    def _fit_time_axis_local(self, frames, min_t=64, max_t=96):
        """
        frames: list or array of [H,W] in [0,1] float32
        returns np.array shaped [T,H,W] with T in [min_t, max_t]
        """
        if isinstance(frames, list):
            if len(frames) == 0:
                return np.zeros((min_t, self.roi_size[0], self.roi_size[1]), dtype=np.float32)
            arr = np.stack(frames, axis=0)
        else:
            arr = frames
        T0 = arr.shape[0]
        if T0 == 0:
            return np.zeros((min_t, self.roi_size[0], self.roi_size[1]), dtype=np.float32)
        if T0 < min_t:
            reps = int(math.ceil(min_t / float(T0)))
            arr = np.repeat(arr, reps, axis=0)[:min_t]
        elif T0 > max_t:
            idx = np.linspace(0, T0 - 1, num=max_t).round().astype(int)
            arr = arr[idx]
        return arr
    # -------------------------------------

    def _extract_mouth_roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0]

        x1, y1, x2, y2 = self._mouth_center_square(lms, w, h, scale=1.0)

        pad_px = max(2, int(min(w, h) * 0.03))
        x1, y1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
        x2, y2 = min(w, x2 + pad_px), min(h, y2 + pad_px)

        x1, y1, x2, y2 = self._smooth_bbox([x1, y1, x2, y2])

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.roi_size, interpolation=cv2.INTER_AREA)

        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            resized = clahe.apply(resized.astype('uint8')).astype('float32')
        except Exception:
            resized = resized.astype('float32')

        resized = resized / 255.0
        return resized.astype(np.float32)

    def _ctc_greedy(self, logits):
        torch, F = self.torch, self.F
        if logits.dim() == 3:
            logits = logits[0]
        probs = F.softmax(logits, dim=-1)
        maxp, pred = probs.max(dim=-1)
        prev = None
        chars, confs = [], []
        for t, p in enumerate(pred.tolist()):
            if p == 0 or p == prev:
                prev = p
                continue
            idx = p if p < len(self.idx2char) else 0
            chars.append(self.idx2char[idx])
            confs.append(float(maxp[t].item()))
            prev = p
        return "".join(chars), (sum(confs) / len(confs) if confs else 0.0)

    def _ctc_beam(self, logits):
        if self.decoder is None:
            return self._ctc_greedy(logits)
        if logits.dim() == 3:
            logits = logits[0]
        lp = self.F.log_softmax(logits, dim=-1).cpu().unsqueeze(0)
        out = self.decoder(lp, self.torch.IntTensor([lp.shape[1]]))
        hyp = out[0][0]
        tokens = hyp.tokens.tolist()
        text = "".join(self.idx2char[i] for i in tokens if i < len(self.idx2char) and i != 0)
        gtext, conf = self._ctc_greedy(logits)
        return (text or gtext), conf

    def predict(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[SilentSpeaker] Could not open video: {video_path}")
            return {"text": "", "conf": 0.0}
        rois = []
        self.prev_bbox = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            roi = self._extract_mouth_roi(frame)
            if roi is not None:
                rois.append(roi)
        cap.release()
        if not rois:
            return {"text": "", "conf": 0.0}
        if self.demo_mode or self.model is None:
            return {"text": "Predicted text from AI model", "conf": 0.0}

        # Fit time axis then prepare tensor
        x = self._fit_time_axis_local(rois)                      # (T,H,W) with T in [64,96]
        x = np.expand_dims(x, 1).astype(np.float32)              # (T,1,H,W)
        xt = self.torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1,T,1,H,W)

        with self.torch.no_grad():
            logits = self.model(xt)                              # (B,T,V)
        if self.use_beam:
            text, conf = self._ctc_beam(logits)
        else:
            text, conf = self._ctc_greedy(logits)
        return {"text": text, "conf": round(float(conf), 3)}
