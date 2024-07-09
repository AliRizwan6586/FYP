from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024  # Set maximum file size to 6 MB

# Load the models
detection_model = tf.keras.models.load_model('citrus_classifier_model.h5')
svm_model = joblib.load('svm_model.pkl')
vgg_model = VGG16(weights='imagenet', include_top=False)

# Define functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features(img_array):
    processed_img = preprocess_input(img_array.copy())
    features = vgg_model.predict(processed_img)
    features_flatten = features.reshape((features.shape[0], -1))
    return features_flatten

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = detection_model.predict(img_array)
    if prediction > 0.5:
        return "Non-Citrus", 0.0
    else:
        features = extract_features(img_array)
        citrus_class = svm_model.predict(features)[0]
        confidence = sigmoid(np.max(svm_model.decision_function(features))) * 100
        return citrus_class, confidence

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Routes
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file extension'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            citrus_class, confidence = predict_image(file_path)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
        finally:
            os.remove(file_path)
        
        if citrus_class == "Non-Citrus":
            return jsonify({'prediction': "Invalid Image!"}), 200
        else:
            return jsonify({'prediction': citrus_class, 'confidence': f"{confidence:.2f}%"})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
