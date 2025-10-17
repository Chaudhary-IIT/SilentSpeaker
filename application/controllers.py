from flask import Blueprint, render_template, request,redirect, url_for
import os
from .models import *
from .database import db
from gtts import gTTS

controllers = Blueprint('controllers', __name__)

# Initialize your AI model
model = LipReaderModel()

@controllers.route('/')
def home():
    return render_template('index.html')

@controllers.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return "No video uploaded", 400

    file = request.files['video']
    path = os.path.join('static/uploads', file.filename)
    file.save(path)

    # Predict text using AI model
    output_text = model.predict(path)

    # Convert to speech
    audio_path = os.path.join('static/uploads', 'output.mp3')
    tts = gTTS(output_text)
    tts.save(audio_path)

    return render_template('result.html', text=output_text, audio_file=audio_path)
