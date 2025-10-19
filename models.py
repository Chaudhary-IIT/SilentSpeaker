# models.py (drop-in replacement)
import cv2, numpy as np
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
        self.lip_indices = list(range(61,88))
        self.prev_bbox = None
        self.smooth_alpha = 0.6
        self.device = device
        self.use_beam = use_beam

        # Resolve ckpt
        base_dir = Path(__file__).parent.resolve()
        default_ckpt = base_dir / "checkpoints" / "lip_reader.pt"
        self.ckpt_path = Path(checkpoint_path).resolve() if checkpoint_path else default_ckpt

        # Defaults
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
                # Load vocab
                if isinstance(ckpt, dict) and "idx2char" in ckpt:
                    self.idx2char = ckpt["idx2char"]
                # Build model and load weights if state_dict is present
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    # Import your training model class
                    from scripts.model_vsr import CNN3D_BiGRU
                    self.model = CNN3D_BiGRU(vocab_size=len(self.idx2char))
                    self.model.load_state_dict(ckpt["state_dict"], strict=False)
                else:
                    # Serialized module or dict["model"]
                    self.model = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.to(self.device).eval()
                # Optional beam decoder
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

    def _landmarks_to_bbox(self, lms, w, h, pad=8):
        xs, ys = [], []
        for i in self.lip_indices:
            lm = lms.landmark[i]
            xs.append(int(lm.x * w)); ys.append(int(lm.y * h))
        x1, y1 = max(min(xs)-pad, 0), max(min(ys)-pad, 0)
        x2, y2 = min(max(xs)+pad, w), min(max(ys)+pad, h)
        return [x1,y1,x2,y2]

    def _smooth_bbox(self, bbox):
        if self.prev_bbox is None:
            self.prev_bbox = bbox; return bbox
        a = self.smooth_alpha
        sm = [int(a*bbox[i] + (1-a)*self.prev_bbox[i]) for i in range(4)]
        self.prev_bbox = sm; return sm

    def _extract_mouth_roi(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lms = res.multi_face_landmarks[0]
        x1,y1,x2,y2 = self._smooth_bbox(self._landmarks_to_bbox(lms, w, h))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0: return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.roi_size, interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32)/255.0)

    def _ctc_greedy(self, logits):
        torch, F = self.torch, self.F
        if logits.dim() == 3: logits = logits[0]      # (T,V)
        probs = F.softmax(logits, dim=-1)
        maxp, pred = probs.max(dim=-1)                # (T,), (T,)
        prev=None; chars=[]; confs=[]
        for t,p in enumerate(pred.tolist()):
            if p==0 or p==prev:
                prev=p; continue
            idx = p if p < len(self.idx2char) else 0
            chars.append(self.idx2char[idx])
            confs.append(float(maxp[t].item()))
            prev=p
        return "".join(chars), (sum(confs)/len(confs) if confs else 0.0)

    def _ctc_beam(self, logits):
        # torchaudio CTCDecoder returns hypotheses; fall back if unavailable
        if self.decoder is None:
            return self._ctc_greedy(logits)
        if logits.dim() == 3: logits = logits[0]   # (T,V)
        # decoder expects log-probs on CPU
        lp = self.F.log_softmax(logits, dim=-1).cpu().unsqueeze(0)  # (1,T,V)
        out = self.decoder(lp, torch.IntTensor([lp.shape[1]]))
        hyp = out[0][0]  # best
        tokens = hyp.tokens.tolist()
        text = "".join(self.idx2char[i] for i in tokens if i < len(self.idx2char) and i != 0)
        # confidence proxy not provided; reuse greedy conf
        gtext, conf = self._ctc_greedy(logits)
        return (text or gtext), conf

    def predict(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[SilentSpeaker] Could not open video: {video_path}")
            return {"text":"", "conf":0.0}
        rois=[]; self.prev_bbox=None
        while True:
            ok, frame = cap.read()
            if not ok: break
            roi = self._extract_mouth_roi(frame)
            if roi is not None: rois.append(roi)
        cap.release()
        if not rois:
            return {"text":"", "conf":0.0}
        if self.demo_mode or self.model is None:
            return {"text":"Predicted text from AI model", "conf":0.0}

        x = np.stack(rois, axis=0).astype(np.float32)     # (T,H,W)
        x = np.expand_dims(x, 1)                          # (T,1,H,W)
        xt = self.torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1,T,1,H,W)
        with self.torch.no_grad():
            logits = self.model(xt)                       # (B,T,V)
        if self.use_beam:
            text, conf = self._ctc_beam(logits)
        else:
            text, conf = self._ctc_greedy(logits)
        return {"text": text, "conf": round(float(conf), 3)}
