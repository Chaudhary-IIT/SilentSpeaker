# 🎙️ SilentSpeaker

SilentSpeaker is an AI-powered web application that can **read lip movements in real time or from a video and convert them into accurate text and speech**.  
It bridges the communication gap for people with **hearing or speech impairments**, and enables silent communication in **noisy, private, or restricted environments**.

---

## 🌟 Key Features

- 🧠 **AI Lip Reading** — Deep learning model interprets lip movements and predicts spoken words.  
- 🎥 **Video Upload or Live Webcam Input** — Users can upload pre-recorded clips or use real-time webcam streaming.  
- 🗣️ **Speech Synthesis** — Converts generated text into human-like audio output.  
- 💬 **Interactive Interface** — Smooth and minimal UI with live feedback display.  
- 📈 **Real-Time Processing** — Low-latency predictions for real-time applications.  
- ☁️ **Flask-Based Backend** — Lightweight, fast, and perfect for integrating AI models.  
- 💻 **Bootstrap Frontend** — Responsive, modern, and easy to use on any device.  
- 🔐 **User Management** — Optional login and history of previous conversions.

---

## 🧩 Tech Stack

| Layer | Technology |
|--------|-------------|
| **Framework** |Streamlit|
| **AI/ML** | PyTorch / TensorFlow (lip-reading model) |
| **Video Processing** | OpenCV, Mediapipe |
| **Speech Synthesis** | gTTS |
| **Database** | SQLite (for user logs and history) |
| **Deployment** | Render / AWS / Hugging Face Spaces |

---

## 🏗️ Project Structure

SilentSpeaker/
│
├── app.py # Flask backend
├── model/
│ ├── lip_reader.py # Deep learning model for lip reading
│ ├── utils.py # Helper functions for preprocessing
│ └── init.py
│
├── static/ # Frontend static assets
│ ├── css/
│ │ └── style.css
│ ├── js/
│ │ └── script.js
│ └── uploads/ # Uploaded videos
│
├── templates/
│ ├── index.html # Main interface
│ ├── result.html # Output display page
│ └── login.html # Optional user authentication
│
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Chaudhary-IIT/SilentSpeaker.git
cd SilentSpeaker    
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
#For Windows
.\venv\Scripts\activate
#For Mac
source venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```


### 4️⃣ Run the app
```bash
python app.py
```


### 5️⃣ Open in browser
```bash
http://127.0.0.1:5000
```

---

## 👥 Contributors

| Name | Role |
| :--- | :--- |
| **Naman Kumar** | Project Architecture, API Development, AI-Web Integration |
| **Yogi Kakadia** | UX/UI Design, Frontend Development, Cross-Browser Compatibility |
| **Ashish** | Data Cleaning, Model Training/Tuning, Model Versioning |
| **Gyan Verma** | Unit/Integration Testing, Bug Tracking, Quality Assurance (QA) |
| **Adity Kumar** | Backend Support, Server Environment Setup, Database Management |
