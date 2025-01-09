print('Importing depndencies')
import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf
import io
import base64
from matplotlib.colors import Normalize

# Initialize Flask appp
from flask import Flask
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
try:
    model_name = 'waste_classification_model.keras'
    print('Loading model: ')
    model = load_model(model_name)  # Update the file name if necessary
    last_conv_layer_name = 'conv2d'  # Replace with your actual last convolutional layer name
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

# Helper function: Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function: Preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    print('Image: ',img_array, img)
    return np.expand_dims(img_array, axis=0), np.array(img)

# Generate Grad-CAM heatmap
def generate_heatmap(model, image):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        grads = tape.gradient(predictions[:, class_idx], conv_outputs)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Overlay heatmap on the original image
def overlay_heatmap(heatmap, img, alpha=0.4):
    cmap = plt.cm.jet
    norm = Normalize(vmin=0, vmax=1)
    heatmap = cmap(norm(heatmap))[:, :, :3]  # Apply colormap
    heatmap = np.uint8(heatmap * 255)
    overlayed_img = heatmap * alpha + img * (1 - alpha)
    return np.uint8(overlayed_img)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received a request to upload a file")

    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("No file selected")
        return jsonify({'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to {filepath}")

        try:
            # Preprocess and analyze the image
            print("Preprocessing the image")
            img_array, img = preprocess_image(filepath)
            print("Running the model prediction")
            prediction = model.predict(img_array)
            print('Inference complete: ')
            # label = 'Recyclable' if prediction[0][0] > 0.5 else 'Non-Recyclable'
            # print(f"Prediction result: {label}")

            # Generate Grad-CAM heatmap
            print("Generating Grad-CAM heatmap")
            heatmap = generate_heatmap(model, img_array)
            heatmap_overlay = overlay_heatmap(heatmap, img)

            # Save the heatmap
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'heatmap.png')
            plt.imsave(heatmap_path, heatmap_overlay)
            print(f"Heatmap saved at {heatmap_path}")

            # Generate LIME explanation
            print("Generating LIME explanation")
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array[0].astype('double'),
                model.predict,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
            lime_img, mask = explanation.get_image_and_mask(
                explanation.top_labels[0],
                positive_only=True,
                num_features=5,
                hide_rest=False
            )
            lime_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lime.png')
            plt.imsave(lime_path, lime_img)
            print(f"LIME image saved at {lime_path}")

            # Encode images as base64 for frontend
            print("Encoding images as base64")
            with open(heatmap_path, "rb") as img_file:
                heatmap_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            with open(lime_path, "rb") as img_file:
                lime_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            print("Returning response with results")
            return jsonify({
                'message': 'Image processed successfully',
                # 'label': label,
                'heatmap': heatmap_base64,
                'lime': lime_base64
            })

        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'message': f'Error processing image: {e}'}), 500

    print("Invalid file format")
    return jsonify({'message': 'Invalid file format'}), 400

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    # Add a mock preprocess_image implementation for debugging purposes
    print(f"Preprocessing image at {filepath}")
    # Simulating a preprocessing step
    return None, None  # Replace with actual preprocessing

def generate_heatmap(model, img_array):
    # Mock generate_heatmap function
    print("Generating mock heatmap")
    return None  # Replace with actual heatmap generation

def overlay_heatmap(heatmap, img):
    # Mock overlay_heatmap function
    print("Overlaying heatmap on image")
    return None  # Replace with actual overlaying

if __name__ == '__main__':
    print("Starting the Flask app")
    app.run(debug=True)
