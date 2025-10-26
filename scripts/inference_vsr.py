import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp

from model_vsr import CNN3D_BiGRU
from vocab import load_vocab, int_to_text

from roi_utils import (
    mouth_center_square as canonical_mouth_center_square,
    add_roi_padding,
)

MIN_T = 75   # minimum frames after fitting
MAX_T = 90   # maximum frames

mp_face_mesh = mp.solutions.face_mesh

def extract_mouth_roi(frame, face_mesh, prev_box=None, smooth_factor=0.6, pad=0.03):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, prev_box
    landmarks = results.multi_face_landmarks[0]
    lip_indices = list(range(61, 89))
    x1, y1, x2, y2 = canonical_mouth_center_square(
        landmarks, w, h, scale=1.2, lip_indices=lip_indices
    )
    x1, y1, x2, y2 = add_roi_padding(x1, y1, x2, y2, w, h, pad_rel=pad)
    if prev_box is not None:
        x1 = int(smooth_factor * prev_box[0] + (1 - smooth_factor) * x1)
        y1 = int(smooth_factor * prev_box[1] + (1 - smooth_factor) * y1)
        x2 = int(smooth_factor * prev_box[2] + (1 - smooth_factor) * x2)
        y2 = int(smooth_factor * prev_box[3] + (1 - smooth_factor) * y2)
    roi = frame[y1:y2, x1:x2]
    return roi, [x1, y1, x2, y2]

def fit_time_axis(frames, min_t=MIN_T, max_t=MAX_T, img_size=(96,96)):
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

def load_model_and_vocab(checkpoint_dir="checkpoints"):
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    ckpt_path = os.path.join(checkpoint_dir, "lip_reader.pt")
    vocab, idx2char = load_vocab(vocab_path)
    model = CNN3D_BiGRU(vocab_size=len(vocab))
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, vocab, idx2char

def greedy_ctc_decode(logits, idx2char):
    pred = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    prev = None
    out = []
    for p in pred:
        if p != prev and p != 0:
            out.append(idx2char[p])
        prev = p
    return "".join(out)

def run_inference(video_frames, model=None, idx2char=None, device="cpu"):
    frames = []
    for f in video_frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    frames = fit_time_axis(frames, MIN_T, MAX_T, (96,96))
    frames = torch.tensor(frames).unsqueeze(0).unsqueeze(2) # (1, T, 1, H, W)
    with torch.no_grad():
        logits = model(frames.to(device))
        decoded = greedy_ctc_decode(logits, idx2char)
    return decoded

def process_video(path, model, idx2char):
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

def preview_rois(path, out_path="preview.mp4"):
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
