import sys
import os
import cv2
import numpy as np
import traceback
from datetime import datetime

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataset_vsr import MouthROIVideoDataset
from scripts.inference_vsr import extract_mouth_roi
import mediapipe as mp

# ====================== SETTINGS ======================
video_idx = 0  # Video index from train_manifest.csv
output_dir = "diagnostics/compare_train_infer"
os.makedirs(output_dir, exist_ok=True)

# Error log setup
error_log_dir = "diagnostics"
os.makedirs(error_log_dir, exist_ok=True)
error_log_path = os.path.join(error_log_dir, "error_log.txt")

report_path = os.path.join(output_dir, "roi_comparison_report.txt")
report_lines = []

def log_error(err_msg, frame_num=None, frame=None):
    """Log error with timestamp, frame number, stack trace, and save problematic frame."""
    with open(error_log_path, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts} | FRAME: {frame_num} | ERROR: {err_msg}\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-"*70 + "\n")
    if frame is not None:
        frame_out = os.path.join(error_log_dir, f"error_frame_{frame_num}.png")
        cv2.imwrite(frame_out, frame)
        print(f"[ERROR] Problematic frame saved: {frame_out}")

# ====================== LOAD TRAINING DATA ======================
try:
    print("[INFO] Loading training dataset ROI...")
    train_ds = MouthROIVideoDataset("data/train_manifest.csv")
    train_roi, _ = train_ds[video_idx]  # shape: (T, 1, H, W)
    train_roi = (train_roi.numpy() * 255).astype(np.uint8)
except Exception as e:
    log_error(f"Failed to load training dataset: {e}")
    raise e

# ====================== INFERENCE ROI EXTRACTION ======================
video_path, _ = train_ds.items[video_idx]
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

cap = cv2.VideoCapture(video_path)
mp_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
prev_box = None
infer_rois = []

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    try:
        roi, prev_box = extract_mouth_roi(frame, mp_mesh, prev_box)
        if roi is not None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (train_roi.shape[2], train_roi.shape[1]))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            infer_rois.append(gray)
    except Exception as e:
        log_error(str(e), frame_num=frame_num, frame=frame)
        print(f"[ERROR] ROI extraction failed for frame {frame_num}: {e}")
    frame_num += 1
cap.release()
infer_rois = np.array(infer_rois)
print(f"[INFO] Got {len(infer_rois)} inference ROIs.")

# ====================== FRAME COUNT CHECK ======================
if len(train_roi) != len(infer_rois):
    warning_msg = f"[WARNING] Frame count mismatch! Train: {len(train_roi)}, Inference: {len(infer_rois)}"
    print(warning_msg)
    report_lines.append(warning_msg)

# ====================== ROI COMPARISON ======================
frame_id = min(len(train_roi), len(infer_rois)) // 2  # middle frame
train_frame = train_roi[frame_id, 0]
infer_frame = infer_rois[frame_id]

# Resize for consistency
h, w = train_frame.shape
infer_frame = cv2.resize(infer_frame, (w, h))

diff_img = cv2.absdiff(train_frame, infer_frame)
compare_img = np.hstack([train_frame, infer_frame, diff_img])
out_path = os.path.join(output_dir, f"roi_compare_{video_idx}.png")
cv2.imwrite(out_path, compare_img)
print(f"[SAVED] ROI comparison image: {out_path}")

# ====================== STATISTICS ======================
mean_diff = np.mean(diff_img)
print(f"[INFO] Mean pixel difference: {mean_diff:.2f}")
report_lines.append(f"Video index: {video_idx}, Mean pixel difference: {mean_diff:.2f}")

if mean_diff > 10:
    warning_msg = "[WARNING] ROI mismatch detected! (BBox / alignment / preprocessing may differ)"
    print(warning_msg)
    report_lines.append(warning_msg)
else:
    ok_msg = "[OK] ROIs look consistent ✅"
    print(ok_msg)
    report_lines.append(ok_msg)

# ====================== SAVE REPORT ======================
with open(report_path, 'w') as f:
    f.write("\n".join(report_lines))
print(f"[SAVED] ROI comparison report: {report_path}")

# ====================== SUMMARY ======================
print(f"[SUMMARY] Total frames processed: {frame_num}, Errors logged: check {error_log_path}")
