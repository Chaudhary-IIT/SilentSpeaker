from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return "No video uploaded", 400

    file = request.files['video']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    # TODO: Replace this with your AI model call
    output_text = "Model output: (Predicted text from lip movements)"

    return render_template('index.html', result=output_text)

if __name__ == "__main__":
    app.run(debug=True)
