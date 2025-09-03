import os
import sys
import traceback
import base64
import io

import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template

# --- Define custom MultiHeadSelfAttention layer ---
# This is necessary for loading a model that uses this custom layer.
class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = keras.layers.Dense(embed_dim)
        self.key_dense = keras.layers.Dense(embed_dim)
        self.value_dense = keras.layers.Dense(embed_dim)
        self.combine_heads = keras.layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

# --- Grad-CAM and Image Utilities ---

def get_last_conv_layer_name(model):
    """Finds the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    return None

def generate_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap and superimposes it on the image."""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    original_img_rgb = np.uint8(img_array[0] * 255)
    superimposed_img = cv2.addWeighted(original_img_rgb, 0.6, heatmap_jet, 0.4, 0)
    
    return superimposed_img

def preprocess_image(image_data):
    """Preprocesses the uploaded image for model prediction."""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def array_to_base64(img_array):
    """Converts a numpy array (image) to a base64 string for HTML display."""
    img = Image.fromarray(img_array.astype('uint8'))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# --- Flask App Setup ---
app = Flask(__name__, template_folder='.')
model = None

def load_model():
    """Loads the Keras model into a global variable."""
    global model
    model_path = 'tumor_model.h5' # Assumes the model is in the same directory
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please place it in the same directory as app.py.")
    
    try:
        with keras.utils.custom_object_scope({'MultiHeadSelfAttention': MultiHeadSelfAttention}):
            model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# Load the model as soon as the script is imported by gunicorn
load_model()

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handles the image prediction request."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    try:
        file = request.files['image']
        image_data = file.read()
        
        processed_image = preprocess_image(image_data)
        
        prediction = model.predict(processed_image, verbose=0)
        
        # Determine prediction result based on model output shape
        if prediction.shape[-1] == 2: # Multi-class [No Tumor, Tumor]
            class_names = ['No Tumor', 'Tumor']
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = float(prediction[0][predicted_class_idx])
        else: # Binary classification
            confidence = float(prediction[0][0])
            predicted_class = 'Tumor' if confidence > 0.5 else 'No Tumor'
        
        print(f"Prediction result: {predicted_class} with confidence {confidence:.2f}")

        result = {
            'predicted_class': predicted_class,
            'confidence': f"{confidence:.2%}",
            'heatmap': None
        }
        
        if predicted_class == 'Tumor':
            print("Predicted class is 'Tumor'. Attempting to generate Grad-CAM heatmap.")
            last_conv_layer_name = get_last_conv_layer_name(model)
            if last_conv_layer_name:
                print(f"Found last conv layer: {last_conv_layer_name}")
                heatmap_img = generate_gradcam_heatmap(model, processed_image, last_conv_layer_name)
                result['heatmap'] = array_to_base64(heatmap_img)
            else:
                print("Could not find a suitable convolutional layer for Grad-CAM.")
                result['error'] = 'Could not find a suitable convolutional layer for Grad-CAM.'

        return jsonify(result)
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
