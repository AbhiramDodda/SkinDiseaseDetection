from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from diseasedirect import diseaesePred
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(image_path):
    result = diseaesePred(image_path)
    return result[0], result[1]

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result, confidence = process_image(filepath)
            return render_template('index.html', filename = filename, result = result, confidence = confidence)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename = 'uploads/' + filename), code = 301)

if __name__ == "__main__":
    app.run(debug=True)
