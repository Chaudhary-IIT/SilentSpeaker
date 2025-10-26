def mouth_center_square(lms, w, h, scale=1.2, lip_indices=None):
    if lip_indices is None:
        lip_indices = list(range(61, 89))
    xs = [int(lms.landmark[i].x * w) for i in lip_indices]
    ys = [int(lms.landmark[i].y * h) for i in lip_indices]
    if len(xs) == 0:
        fb = _face_bbox_from_landmarks(lms, w, h)
        fw = fb[2] - fb[0]
        cx = fb[0] + fw // 2
        cy = fb[1] + (fb[3] - fb[1]) // 2
        size = int(fw * 0.38)
        half = size // 2
        return (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))
    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))
    lip_w = max(1.0, max(xs) - min(xs))
    face_bbox = _face_bbox_from_landmarks(lms, w, h)
    face_w = float(face_bbox[2] - face_bbox[0])
    size = int(max(lip_w * 2.8, face_w * 0.30))
    size = int(size * float(scale))
    half = max(8, size // 2)
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return (x1, y1, x2, y2)

def _face_bbox_from_landmarks(lms, w, h, pad_rel=0.08):
    xs = [int(lm.x * w) for lm in lms.landmark]
    ys = [int(lm.y * h) for lm in lms.landmark]
    x1, y1 = max(min(xs), 0), max(min(ys), 0)
    x2, y2 = min(max(xs), w), min(max(ys), h)
    pad_x = int((x2 - x1) * pad_rel)
    pad_y = int((y2 - y1) * pad_rel)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y))

def add_roi_padding(x1, y1, x2, y2, w, h, pad_rel=0.03):
    pad_px = max(2, int(min(w, h) * pad_rel))
    x1, y1 = max(0, x1 - pad_px), max(0, y1 - pad_px)
    x2, y2 = min(w, x2 + pad_px), min(h, y2 + pad_px)
    return (x1, y1, x2, y2)
