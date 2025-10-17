from .database import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    history = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500))
    audio_path = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))



class LipReaderModel:
    def __init__(self):
        # Load your trained PyTorch/TensorFlow model here
        pass

    def predict(self, video_path):
        # Process video and return predicted text
        return "Predicted text from AI model"
