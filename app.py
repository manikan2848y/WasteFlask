'''import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Define the folder for uploaded images
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Recyclable vs non-recyclable mapping
RECYCLABLE_CLASSES = {
    "shovel":True,
    "paper_towel":True,
    "packet":True,
    "envelope":True,
    "crossword_puzzle":True,
    "wine_bottle": True,
    "plastic_bottle": False,
    "glass_bottle": True,
    "cardboard": True,
    "carton":True,
    "metal_can": True,
    "paper": True,
    "plastic_bag": False,
    "food_waste": False,
    "electronics": False,
    "clothing": False,
}
# shovel", "crossword_puzzle" , "packet
# Disable TensorFlow optimizations that may not be supported on your hardware
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Try loading the model and handle errors
def load_model():
    try:
        model = DenseNet121(weights='imagenet')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the pre-trained DenseNet121 model
model = load_model()
input_size = (224, 224)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=input_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_class(img_path):
    try:
        img_array = preprocess_image(img_path)
        print(img_array.shape)
        predictions = model.predict(img_array)
        
        print(predictions.shape)
        decoded_preds = decode_predictions(predictions, top=1)[0]
        predicted_class, class_name, confidence = decoded_preds[0]
        return class_name, confidence
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")

# Function to check recyclability
def is_recyclable(class_name):
    if class_name in RECYCLABLE_CLASSES:
        return RECYCLABLE_CLASSES[class_name]
    else:
        print(f"Warning: Unknown class name '{class_name}' encountered. Defaulting to False.")
        return False

def apply_grad_cam(img_path, class_name):
    try:
        print(f"Applying Grad-CAM for class: {class_name}")
        
        # Preprocess the input image
        img_array = preprocess_image(img_path)
        print(f"Image array shape for Grad-CAM: {img_array.shape}")
        
        # Convert to tf.Tensor
        img_tensor = tf.convert_to_tensor(img_array)
        print(f"Converted image array to Tensor: {img_tensor.shape}, dtype: {img_tensor.dtype}")

        # Define the gradient model
        grad_model = tf.keras.Model(
            inputs=model.input,  # Ensure the input is connected properly
            outputs=[
                model.get_layer("conv5_block16_2_conv").output,  # Convolutional layer for Grad-CAM
                model.output,  # Final prediction output
            ]
        )
        print("Grad-CAM model initialized successfully.")

        # Compute the gradient of the predicted class with respect to the output
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)  # Watch the image tensor
            conv_outputs, predictions = grad_model(img_tensor)
            predicted_class_index = np.argmax(predictions[0])  # Get the index of the predicted class
            loss = predictions[:, predicted_class_index]  # Focus on the predicted class
            print(f"Predicted class index: {predicted_class_index}, Loss: {loss}")

        # Compute the gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling for gradients
        print(f"Pooled gradients shape: {pooled_grads.shape}")

        # Generate the Grad-CAM heatmap
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # Normalize
        print("Grad-CAM heatmap generated successfully.")
        return heatmap.numpy()
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        raise


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    try:
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return overlay
    except Exception as e:
        raise RuntimeError(f"Error overlaying heatmap: {e}")

def save_uploaded_file(file):
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File successfully saved at {filepath}")  # Debug log
        return filepath
    except Exception as e:
        print(f"Error saving the uploaded file: {e}")  # Debug log
        raise ValueError(f"Error saving the uploaded file: {e}")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request to /predict endpoint.")  # Log entry point

    # Check if the file part is in the request
    if 'file' not in request.files:
        print("Error: No file part in the request.")  # Log missing file part
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        print("Error: No file selected for upload.")  # Log no file selected
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file
        print(f"Saving uploaded file: {file.filename}")  # Log file saving
        filepath = save_uploaded_file(file)
        print(f"File saved successfully at: {filepath}")  # Log file path

        # Predict the class and confidence
        print("Starting prediction process...")  # Log prediction start
        class_name, confidence = predict_class(filepath)
        print(f"Prediction complete. Class: {class_name}, Confidence: {confidence}")  # Log prediction results

        # Check if the item is recyclable
        print(f"Checking recyclability for class: {class_name}")  # Log recyclability check
        recyclable = is_recyclable(class_name)
        print(f"Recyclable: {recyclable}")  # Log recyclability result

        # Generate Grad-CAM heatmap
        print("Generating Grad-CAM heatmap...")  # Log heatmap generation start
        heatmap = apply_grad_cam(filepath, class_name)
        print("Grad-CAM heatmap generated successfully.")  # Log heatmap success

        # Overlay the heatmap on the original image
        print("Overlaying heatmap on the original image...")  # Log overlay process
        overlayed_img = overlay_heatmap(filepath, heatmap)
        overlayed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'overlayed_img.jpg')
        cv2.imwrite(overlayed_img_path, overlayed_img)
        print(f"Overlayed image saved at: {overlayed_img_path}")  # Log overlayed image path

        # Prepare JSON response
        print("Preparing JSON response...")  # Log response preparation
        response = {
            'class_name': class_name,
            'confidence': float(round(confidence, 2)),  # Convert confidence to Python float
            'recyclable': recyclable,
            'overlayed_img_url': f'/uploads/{os.path.basename(overlayed_img_path)}'
        }
        print(f"Response: {response}")  # Log response
        return jsonify(response)

    except Exception as e:
        # Log any errors that occur during the process
        print(f"Error analyzing the file: {e}")  # Log error message
        return jsonify({'error': f"Error analyzing the file: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

'''


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the pre-trained DenseNet model
@st.cache_resource
def load_model():
    return tf.keras.applications.DenseNet121(weights='imagenet')

model = load_model()

# Recyclable vs Non-recyclable classes
RECYCLABLE_CLASSES = {
    "shovel": True, "paper_towel": True, "packet": True, "envelope": True, "crossword_puzzle": True,
    "wine_bottle": True, "plastic_bottle": False, "glass_bottle": True, "cardboard": True,
    "carton": True, "metal_can": True, "paper": True, "plastic_bag": False, "food_waste": False,
    "electronics": False, "clothing": False,
}

# Image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Prediction and recyclability
def predict_and_check(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=1)[0]
    class_name, confidence = decoded_preds[0][1], decoded_preds[0][2]
    recyclable = RECYCLABLE_CLASSES.get(class_name, False)
    return class_name, confidence, recyclable

# Grad-CAM for explainability
def apply_grad_cam(img_path):
    img_array = preprocess_image(img_path)
    grad_model = tf.keras.Model(
        [model.input], [model.get_layer("conv5_block16_2_conv").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    img = cv2.imread(original_img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img

# Streamlit UI
st.sidebar.title("Waste Classification with Explainable AI")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "CLASSIFICATION & EXPLAINABILITY"])

if app_mode == "HOME":
    st.title("Waste Classification with Explainable AI")
   # st.image("C:/Users/saisa/Plant_disease_new/Diseases.png", use_column_width=True)  # Updated path to your image
    st.image("C:/Users/saisa/Plant_disease_new/Diseases.png", caption="Use the Waste", use_container_width=True)  # Updated parameter

    st.write("""
    This application helps classify waste items and provides insights using Grad-CAM 
    for explainability. Upload an image to see predictions and recyclability information.
    """)

elif app_mode == "CLASSIFICATION & EXPLAINABILITY":
    st.header("Waste Classification")
    uploaded_image = st.file_uploader("Upload a waste image:", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img_path = "uploaded_image.jpg"
        img = Image.open(uploaded_image)
        img.save(img_path)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            st.write("Classifying...")
            class_name, confidence, recyclable = predict_and_check(img_path)
            st.success(f"Class: {class_name} | Confidence: {confidence:.2f}")
            st.info(f"Recyclable: {'Yes' if recyclable else 'No'}")

        if st.button("Explain with Grad-CAM"):
            st.write("Generating Grad-CAM visualization...")
            heatmap = apply_grad_cam(img_path)
            overlayed_img = overlay_heatmap(img_path, heatmap)
            st.image(overlayed_img, caption="Grad-CAM Visualization", use_column_width=True)

