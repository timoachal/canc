import os
import sys
import traceback
import base64
import io

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

import tensorflow as tf
from tensorflow import keras

# --- Flask App Setup ---
app = Flask(__name__, template_folder='.')
interpreter = None
input_details = None
output_details = None
class_names = ['No Tumor', 'Tumor']

def load_model():
    """Loads the quantized TFLite model into a global variable."""
    global interpreter, input_details, output_details
    model_path = 'quantized_tumor_model.tflite'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please ensure it is in the same directory.")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("TFLite model loaded successfully!")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        raise e

def preprocess_image(image_data, input_shape):
    """Preprocesses the uploaded image for TFLite model prediction."""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        height, width = input_shape[1], input_shape[2]
        image = image.resize((width, height))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

# Load the model as soon as the script is imported by gunicorn
load_model()

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handles the image prediction request."""
    print(f"Prediction request received from URL: {request.url}")
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    try:
        file = request.files['image']
        image_data = file.read()

        # Get the required input shape from the interpreter
        input_shape = input_details[0]['shape']
        processed_image = preprocess_image(image_data, input_shape)

        # Set the tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        
        # Get the output tensor
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Get the index of the highest confidence score
        predicted_index = np.argmax(prediction)
        
        # Get the confidence of the predicted class
        confidence = float(prediction[0][predicted_index])
        
        predicted_class = class_names[predicted_index]
        
        print(f"Prediction result: {predicted_class} with confidence {confidence:.2f}")

        # Determine recommendation based on prediction
        if predicted_class == 'Tumor':
            recommendation = "This result suggests the presence of a tumor. Please consult with a medical professional for an accurate diagnosis and treatment plan."
        else:
            recommendation = "This result suggests no tumor is present. However, this is a preliminary finding from a machine learning model and does not replace professional medical advice. Please consult with a doctor for a definitive diagnosis."

        result = {
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2%}",
            'recommendation': recommendation,
            'heatmap': None # Grad-CAM is not available for TFLite models
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
