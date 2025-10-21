# SilentSpeaker: Visual Speech Recognition Web App

**SilentSpeaker** is an AI web application that translates lip movements into text and optional speech. It's designed for **assistive communication** and **silent interaction** in environments where microphones are unusable, such as noisy public spaces or private settings.

It operates on both uploaded videos and live webcam streams through a clean, interactive **Streamlit** interface.

## 🚀 Highlights

* **Visual Speech Recognition (VSR):** Uses a **CTC-based** model to decode character sequences directly from mouth-region video frames, operating completely **without audio**.
* **Dual Input Modes:** Supports **video file upload** and **live webcam** streaming with an optional landmark overlay for transparent debugging of the **Region of Interest (ROI)**.
* **Optional Speech Output:** Includes lightweight **Text-to-Speech (TTS)** generation (`gTTS`) and a persistent, one-click **History** that locally saves transcripts, translations, and media paths.

---

## 💻 Tech Stack

| Layer | Technology | 
| :--- | :--- | 
| **Frontend/UX** | Streamlit | 
| **Real-time I/O** | `streamlit-webrtc` | 
| **Vision** | MediaPipe Face Mesh, OpenCV | 
| **AI/ML** | PyTorch model + CTC decoding | 
| **Speech** | `gTTS` | 
| **Storage** | SQLite + CSV log | 
| **Deployment** | Hugging Face Spaces (containerized) or any VM with Python | 

---

## 🏗️ System Overview

* **Frontend/UX:** Implemented with Streamlit tabs for **Upload**, **Webcam**, and **History**, ensuring a consistent data flow: `predict` → `translate` → `TTS` → `persist`.
* **Vision:** **MediaPipe Face Mesh** landmarks are used to isolate a stable **mouth ROI** per frame. These frames are normalized and stacked into sequences for the model.
* **Model:** A compact **3D-CNN + BiGRU** network is trained with **CTC (Connectionist Temporal Classification)**. It uses **greedy decoding** by default, with potential for beam decoding in the roadmap.
* **Persistence:** A **SQLite** database (`history.db`) is used for the main history, backed up by a **CSV log** (`history.csv`). Media (videos, audio) are stored under `static/uploads` and `static/audio`.

---

## 📂 Repository Layout

```text
SilentSpeaker/
├─ app.py                        # Streamlit app (Upload, Webcam, History)
├─ models.py                     # Inference wrapper (ROI, ckpt load, CTC decode)
├─ models/
│  └─ checkpoints/               # Place lip_reader.pt here
├─ scripts/
│  ├─ build_manifest_grid_v3.py  # Pair videos with alignments → CSV manifest
│  ├─ dataset_vsr.py             # ROI dataset + collate for CTC
│  ├─ model_vsr.py               # CNN3D + BiGRU model
│  ├─ train_vsr.py               # CTC training loop (CPU or CUDA auto)
│  └─ vocab.py                   # CTC vocab utilities
├─ static/
│  ├─ uploads/                   # Saved videos (upload/webcam)
│  ├─ audio/                     # Generated TTS audio
│  ├─ history.db                 # SQLite history
│  └─ history.csv                # CSV backup log
├─ requirements.txt
└─ README.md
```
---

---

## ⚙️ Installation

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Quick Start (Inference Only)

1.  **Obtain a Checkpoint:** Place a trained model checkpoint at the following location:
    ```text
    models/checkpoints/lip_reader.pt
    ```

2.  **Launch the App:**
    ```bash
    streamlit run app.py
    ```

3.  **Load Weights:** In the Streamlit sidebar, click **“Reload model”** once to ensure the new weights are loaded from disk.

4.  **Use the App:**
    * **Upload Tab:** Pick a short video clip. The output will show **Predicted Text**, confidence, translation, and play audio. A new row will appear in the History tab.
    * **Webcam Tab:** Start the camera, speak silently for about 3–5 seconds, then click **“Save Transcript.”** A confirmation toast and a new row in History will confirm persistence.

> **Note:** The "Stop" button's auto-save functionality can be browser-dependent. The **“Save Transcript”** button is the guaranteed path to persistence.

---

## 📚 Data Preparation (GRID-style Dataset)

This setup assumes a **GRID-style** dataset structure (e.g., a Kaggle set) where videos and alignment files are separated.

Assuming your dataset root is structured like this:

```text
...\data
├─ s1\*.mpg
└─ alignments\s1\*.align
```

Build the training and validation manifests with absolute paths:

```bash
python scripts/build_manifest_grid_v3.py --root "C:\path\to\...\lipreading-dataset\versions\1\data" --outdir data
```

This command produces:
```text
data/train_manifest.csv
data/val_manifest.csv
```

Each CSV row will contain the video path and the corresponding ground truth text:
```csv
video_path,text
C:\...\s1\bbaf2n.mpg,bin green at nine please
```

---

## 🧠 Train the Model

Run the CTC training script using the generated manifests. The script automatically selects between **CPU** and **CUDA** based on availability.

```bash
python scripts/train_vsr.py data/train_manifest.csv data/val_manifest.csv
```

Upon improvement in the validation loss, the script saves the best model:
```text
checkpoints/lip_reader.pt
```

After training, copy or move that file to the application's expected location:
```text
models/checkpoints/lip_reader.pt
```

Then, click “Reload model” in the app sidebar.

---

## 🎯 Design Choices

* **Face Mesh for ROI:** Ensures a stable and robust **mouth ROI isolation**, which makes the pipeline reliable and lightweight for real-time operation.
* **CTC Decoding:** Allows for efficient **sentence-level decoding** without the need for time-consuming, per-frame label alignment. **Greedy search** provides efficient real-time performance.
* **Streamlit & WebRTC:** Chosen for **rapid iteration**, **accessible deployment**, and a user-friendly demo surface that non-technical users can interact with easily.

---

## 🗺️ Roadmap

* Improve decoding by integrating a **beam search decoder** with an optional **language model** for higher accuracy.
* Expand training using multi-speaker sentence datasets and add **augmentation** for better generalization across different pose and lighting conditions.
* Offer a **higher-fidelity TTS** as an optional backend while retaining the lightweight default.
* Package the application with a **Dockerfile** for one-click, containerized deployment on Hugging Face Spaces or any cloud VM.

---

## 🩹 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Placeholder text only** | No checkpoint detected. Place `lip_reader.pt` under `models/checkpoints` and use the **“Reload model”** button in the sidebar. |
| **History empty on webcam Stop** | Use the **“Save Transcript”** button instead. The "Stop" auto-flush relies on the specific browser state and may not be reliable. |
| **CPU-only error about CUDA** | The training script automatically falls back to CPU if CUDA is not available. Performance will be slower, but fully functional. |

---

## 👥 Team

| Name | Role |
| :--- | :--- |
| **Naman Kumar** | Architecture, model-app integration, end-to-end demo |
| **Gyan Verma** | Validation, QA, test procedures, UI/UX Design |
| **Adity Kumar** | Environment, packaging, local data handling, training/tuning |
| **Yogi Kakadia** | Debugging |
| **Ashish** | Debugging |

---

## 📜 License

BABA GROUP (IIT Mandi)
