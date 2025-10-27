import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from scripts.model_vsr import CNN3D_BiGRU
from scripts.vocab import load_vocab, int_to_text



# ==================== ROI Extraction ====================

mp_face_mesh = mp.solutions.face_mesh

def extract_mouth_roi(frame, face_mesh, prev_box=None, smooth_factor=0.7, pad=0.1):
    """
    Extract a tight, stable mouth-centered ROI from a frame using MediaPipe FaceMesh.
    Returns (roi_frame, bbox) or (None, prev_box) if detection fails.
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, prev_box

    landmarks = results.multi_face_landmarks[0]
    # mouth landmarks: indices 61–88
    mouth_pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark[61:89]])
    x_min, y_min = mouth_pts.min(axis=0)
    x_max, y_max = mouth_pts.max(axis=0)

    # expand slightly but keep tight focus
    box_w, box_h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    scale = 0.8
    box_w *= scale * (1 + pad)
    box_h *= scale * (1 + pad)
    x_min = int(max(cx - box_w / 2, 0))
    y_min = int(max(cy - box_h / 2, 0))
    x_max = int(min(cx + box_w / 2, w))
    y_max = int(min(cy + box_h / 2, h))

    # temporal smoothing
    if prev_box is not None:
        x_min = int(smooth_factor * prev_box[0] + (1 - smooth_factor) * x_min)
        y_min = int(smooth_factor * prev_box[1] + (1 - smooth_factor) * y_min)
        x_max = int(smooth_factor * prev_box[2] + (1 - smooth_factor) * x_max)
        y_max = int(smooth_factor * prev_box[3] + (1 - smooth_factor) * y_max)

    roi = frame[y_min:y_max, x_min:x_max]
    return roi, [x_min, y_min, x_max, y_max]


# ==================== Model / Vocab Loading ====================

def load_model_and_vocab(checkpoint_dir="checkpoints"):
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    ckpt_path = os.path.join(checkpoint_dir, "lip_reader.pt")

    vocab, idx2char = load_vocab(vocab_path)
    model = CNN3D_BiGRU(vocab_size=len(vocab))

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # ✅ Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # assume raw state_dict

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, vocab, idx2char


# ==================== Decoding ====================

def greedy_ctc_decode(logits, idx2char):
    """Greedy CTC decoding"""
    pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    prev = None
    out = []
    for p in pred:
        if p != prev and p != 0:  # skip blanks and repeats
            out.append(idx2char[p])
        prev = p
    return "".join(out)


# ==================== Inference ====================

def run_inference(video_frames, model=None, idx2char=None, device="cpu"):
    """
    Run inference on a list of ROI frames (already cropped around mouth).
    """
    frames = []
    for f in video_frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96))
        frames.append(gray)

    frames = np.array(frames).astype("float32") / 255.0
    frames = torch.tensor(frames).unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)

    with torch.no_grad():
        logits = model(frames.to(device))
        decoded = greedy_ctc_decode(logits, idx2char)
    return decoded


# ==================== Process Video File ====================

def process_video(path, model, idx2char):
    """
    Opens a video file, extracts mouth ROIs, runs model inference, returns predicted text.
    """
    cap = cv2.VideoCapture(path)
    mp_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    prev_box = None
    rois = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        roi, prev_box = extract_mouth_roi(frame, mp_mesh, prev_box)
        if roi is not None:
            rois.append(roi)
    cap.release()

    if len(rois) == 0:
        return "[No face detected]"
    return run_inference(rois, model, idx2char)


# ==================== Debug ROI Preview ====================

def preview_rois(path, out_path="preview.mp4"):
    """
    Generates a video showing detected mouth ROI boxes for visual debugging.
    """
    cap = cv2.VideoCapture(path)
    mp_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    prev_box = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        roi, prev_box = extract_mouth_roi(frame, mp_mesh, prev_box)
        if roi is not None:
            x1, y1, x2, y2 = prev_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if out is None:
            h, w, _ = frame.shape
            out = cv2.VideoWriter(out_path, fourcc, 25, (w, h))
        out.write(frame)
    cap.release()
    if out is not None:
        out.release()
    print(f"[INFO] ROI preview saved to {out_path}")


# ==================== CLI Entry ====================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", default="checkpoints", help="Path to checkpoint folder")
    args = parser.parse_args()

    model, vocab, idx2char = load_model_and_vocab(args.checkpoint)
    print(f"Running inference on {args.video}...")
    text = process_video(args.video, model, idx2char)
    print("Predicted Text:", text)
