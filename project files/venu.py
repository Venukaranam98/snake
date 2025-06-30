from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained Keras model
model = tf.keras.models.load_model('cnn.keras')

# Class names
class_names = [
    'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena', 'combretum', 'croton',
    'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea', 'matayba', 'mimosa',
    'myrcia', 'protium', 'qualea', 'schinus', 'senegalia', 'serjania', 'syagrus',
    'tridax', 'urochloa', 'anadenanthera'
]

# Constants
IMAGE_SIZE = 128

# Prediction function
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            predicted_class, confidence = predict(img)

            return render_template(
                'prediction.html',
                 image_path=filepath,
                predicted_label=predicted_class,
                confidence=confidence
                )
        else:
            return render_template('index.html', message='Invalid file type. Only JPG, JPEG, PNG allowed.')

    return render_template('index.html', message='Upload an image')

# Check for allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
