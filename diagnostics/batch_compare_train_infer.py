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
output_dir = "diagnostics/batch_compare_train_infer"
os.makedirs(output_dir, exist_ok=True)

error_log_dir = "diagnostics"
os.makedirs(error_log_dir, exist_ok=True)
error_log_path = os.path.join(error_log_dir, "error_log.txt")

summary_report_path = os.path.join(output_dir, "batch_roi_comparison_summary.txt")
summary_lines = []

# Pixel difference threshold to flag mismatch
PIXEL_DIFF_THRESHOLD = 10

# ====================== ERROR LOGGING FUNCTION ======================
def log_error(err_msg, video_idx=None, frame_num=None, frame=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_log_path, "a") as f:
        f.write(f"{ts} | VIDEO: {video_idx} | FRAME: {frame_num} | ERROR: {err_msg}\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-"*70 + "\n")
    if frame is not None:
        frame_out = os.path.join(error_log_dir, f"error_video{video_idx}_frame{frame_num}.png")
        cv2.imwrite(frame_out, frame)
        print(f"[ERROR] Problematic frame saved: {frame_out}")

# ====================== LOAD DATASET ======================
try:
    print("[INFO] Loading training dataset...")
    train_ds = MouthROIVideoDataset("data/train_manifest.csv")
except Exception as e:
    log_error(f"Failed to load dataset: {e}")
    raise e

# ====================== BATCH PROCESSING ======================
for video_idx, (video_path, _) in enumerate(train_ds.items):
    try:
        print(f"[INFO] Processing video {video_idx}: {video_path}")
        # Load training ROI
        train_roi, _ = train_ds[video_idx]
        train_roi = (train_roi.numpy() * 255).astype(np.uint8)
        T_train = len(train_roi)

        # Inference ROI extraction
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
                log_error(str(e), video_idx=video_idx, frame_num=frame_num, frame=frame)
                print(f"[ERROR] ROI extraction failed for video {video_idx}, frame {frame_num}: {e}")
            frame_num += 1
        cap.release()
        infer_rois = np.array(infer_rois)
        T_infer = len(infer_rois)
        print(f"[INFO] Video {video_idx}: Training frames={T_train}, Inference frames={T_infer}")

        # Frame count mismatch warning
        if T_train != T_infer:
            warning_msg = f"[WARNING] Frame count mismatch! Video {video_idx}: Train={T_train}, Infer={T_infer}"
            print(warning_msg)
            summary_lines.append(warning_msg)

        # ROI comparison (middle frame)
        if T_train > 0 and T_infer > 0:
            frame_id = min(T_train, T_infer) // 2
            train_frame = train_roi[frame_id, 0]
            infer_frame = infer_rois[frame_id]
            h, w = train_frame.shape
            infer_frame = cv2.resize(infer_frame, (w, h))
            diff_img = cv2.absdiff(train_frame, infer_frame)

            # Save comparison image
            compare_img = np.hstack([train_frame, infer_frame, diff_img])
            out_path = os.path.join(output_dir, f"roi_compare_{video_idx}.png")
            cv2.imwrite(out_path, compare_img)
            print(f"[SAVED] ROI comparison image: {out_path}")

            mean_diff = np.mean(diff_img)
            summary_line = f"Video {video_idx}: Mean pixel difference = {mean_diff:.2f}"
            if mean_diff > PIXEL_DIFF_THRESHOLD:
                summary_line += " [WARNING: ROI mismatch]"
            print(f"[INFO] {summary_line}")
            summary_lines.append(summary_line)
        else:
            msg = f"[INFO] Video {video_idx}: No frames available for comparison"
            print(msg)
            summary_lines.append(msg)

    except Exception as e_video:
        log_error(str(e_video), video_idx=video_idx)
        print(f"[ERROR] Failed to process video {video_idx}: {e_video}")
    print("-" * 70)

# ====================== SAVE SUMMARY REPORT ======================
with open(summary_report_path, 'w') as f:
    f.write("\n".join(summary_lines))
print(f"[SAVED] Batch ROI comparison summary: {summary_report_path}")
print(f"[INFO] Total videos processed: {len(train_ds.items)}. Check {error_log_path} for errors.")
