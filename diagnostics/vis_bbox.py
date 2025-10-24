from models import LipReaderModel
import cv2, os
m = LipReaderModel(checkpoint_path="models/checkpoints/lip_reader.pt", device="cpu")
cap = cv2.VideoCapture("static/uploads/0d2881fbcea7468a8c0a9cc4eb58d4f7_bbwtzp.mpg")
outdir = "diagnostics/annot"
os.makedirs(outdir, exist_ok=True)
i=0
while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = m.face_mesh.process(rgb)
    if res.multi_face_landmarks:
        lms = res.multi_face_landmarks[0]
        # compute mouth bbox using new helper
        x1,y1,x2,y2 = m._mouth_center_square(lms, frame.shape[1], frame.shape[0], scale=1.0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(f"{outdir}/frame_{i:04d}.jpg", frame)
    i+=1
cap.release()
print("saved", i, "frames to", outdir)
