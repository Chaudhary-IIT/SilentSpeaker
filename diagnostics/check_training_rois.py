import sys
import os
import cv2
import numpy as np
import traceback
from datetime import datetime

# Add project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataset_vsr import MouthROIVideoDataset

# ====================== SETTINGS ======================
output_dir = 'diagnostics/train_roi_frames'
os.makedirs(output_dir, exist_ok=True)

# Error log setup
error_log_dir = "diagnostics"
os.makedirs(error_log_dir, exist_ok=True)
error_log_path = os.path.join(error_log_dir, "error_log.txt")

def log_error(err_msg, sample_idx=None, frame_pos=None, frame=None):
    """Log error with timestamp, sample index, frame position, stack trace, and save frame."""
    with open(error_log_path, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts} | SAMPLE: {sample_idx} | FRAME_POS: {frame_pos} | ERROR: {err_msg}\n")
        f.write(traceback.format_exc())
        f.write("\n" + "-"*70 + "\n")
    if frame is not None:
        frame_out = os.path.join(error_log_dir, f"error_sample{sample_idx}_frame_{frame_pos}.png")
        cv2.imwrite(frame_out, frame)
        print(f"[ERROR] Problematic frame saved: {frame_out}")

# ====================== LOAD DATASET ======================
try:
    print("[INFO] Loading training dataset...")
    ds = MouthROIVideoDataset("data/train_manifest.csv")
except Exception as e:
    log_error(f"Failed to load dataset: {e}")
    raise e

# ====================== EXTRACT & SAVE ROI FRAMES ======================
for i in range(5):
    try:
        x, y = ds[i]  # x shape: (T, 1, H, W)
        print(f"[INFO] Sample {i} label/text: {y}")

        T = x.shape[0]
        idxs = [0, T // 2, T - 1]  # first, middle, last frames

        for pos, idx in zip(['first', 'middle', 'last'], idxs):
            try:
                frame = x[idx, 0].numpy()  # H, W
                frame_img = (frame * 255).astype(np.uint8)

                # Apply CLAHE for consistency
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                frame_clahe = clahe.apply(frame_img)

                out_path = os.path.join(output_dir, f'roi_{i}_{pos}.png')
                cv2.imwrite(out_path, frame_clahe)
                print(f"[SAVED] {out_path}")
            except Exception as e_frame:
                log_error(str(e_frame), sample_idx=i, frame_pos=pos, frame=frame_img)
                print(f"[ERROR] Failed to process frame {pos} of sample {i}: {e_frame}")
    except Exception as e_sample:
        log_error(str(e_sample), sample_idx=i)
        print(f"[ERROR] Failed to process sample {i}: {e_sample}")
    print("-" * 50)

print("[INFO] ROI frame extraction complete. Check logs for any errors.")
