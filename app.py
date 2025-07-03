import os
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Set upload folder inside 'static/' so Flask can serve the image
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('best_model.h5')

# Define your class labels
class_labels = ['animal', 'cartoon', 'floral', 'geometry', 'ikat',
                'plain', 'polka dot', 'squares', 'stripes', 'tribal']

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# File upload and classification route
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if file is in request
    if 'file' not in request.files:
        return redirect(request.url)  # If no file is in the request, redirect back
    
    file = request.files['file']
    
    # If the file is valid (not empty and correct extension)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure the upload folder exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Save the file to the uploads folder
        file.save(file_path)

        # Preprocess the image for model prediction
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Model prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        # Fix file path for template rendering
        relative_path = os.path.join('uploads', filename).replace('\\', '/')

        # Return the result page with image and prediction
        return render_template('page3.html', image_url=relative_path, prediction=predicted_label)

    # If no valid file, redirect to index page
    return redirect(url_for('index'))

# Result page (optional)
@app.route('/result')
def result():
    return render_template('result.html')

# Input-output page (optional)
@app.route('/input_output')
def input_output():
    return render_template('input_output.html')

if __name__ == '__main__':
    # Ensure the upload folder exists inside 'static/'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
