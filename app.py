from flask import Flask
from application.database import db
from application.controllers import controllers

app=None

def create_app():
    app = Flask(__name__)
    app.register_blueprint(controllers)
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///silent_speaker.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
