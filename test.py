import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

# Disable TensorFlow optimizations that may not be supported on your hardware
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Recyclable vs non-recyclable mapping
RECYCLABLE_CLASSES = {
    "plastic_bottle": True,
    "glass_bottle": True,
    "cardboard": True,
    "metal_can": True,
    "paper": True,
    "plastic_bag": False,
    "food_waste": False,
    "electronics": False,
    "clothing": False,
}

# Try loading the model and handle errors
def load_model():
    try:
        print("Loading DenseNet121 model...")
        model = DenseNet121(weights='imagenet')
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the pre-trained DenseNet121 model
model = load_model()
input_size = (224, 224)

# Preprocess the image
def preprocess_image(img_path):
    try:
        print(f"Preprocessing image: {img_path}")
        img = image.load_img(img_path, target_size=input_size)
        img_array = image.img_to_array(img)
        print(f"Image array shape before expansion: {img_array.shape}")
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Image array shape after expansion: {img_array.shape}")
        img_array = preprocess_input(img_array)
        print("Image preprocessing complete.")
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise

# Predict the class of the image
def predict_class(img_path):
    try:
        img_array = preprocess_image(img_path)
        print("Predicting class...")
        predictions = model.predict(img_array)
        print(f"Raw predictions: {predictions}")
        decoded_preds = decode_predictions(predictions, top=1)[0]
        print(f"Decoded predictions: {decoded_preds}")
        predicted_class, class_name, confidence = decoded_preds[0]
        print(f"Predicted class: {class_name}, Confidence: {confidence}")
        return class_name, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

# Check if the class is recyclable
def is_recyclable(class_name):
    recyclable = RECYCLABLE_CLASSES.get(class_name, False)
    print(f"Class '{class_name}' is recyclable: {recyclable}")
    return recyclable

# Apply Grad-CAM to generate heatmap
def apply_grad_cam(img_path, class_name):
    try:
        print("Applying Grad-CAM...")
        img_array = preprocess_image(img_path)
        grad_model = tf.keras.Model(
            [model.inputs],
            [model.get_layer("conv5_block16_2_conv").output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, np.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        print("Grad-CAM heatmap generated.")
        return heatmap.numpy()
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        raise

# Overlay heatmap on image
def overlay_heatmap(img_path, heatmap, alpha=0.4):
    try:
        print("Overlaying heatmap on image...")
        img = cv2.imread(img_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        print("Heatmap overlay complete.")
        return overlay
    except Exception as e:
        print(f"Error overlaying heatmap: {e}")
        raise

# Test the pipeline with a dummy image
if __name__ == "__main__":
    dummy_image_path = "uploads\Metal_32.jpg"  # Replace with a valid image path
    try:
        print("Starting prediction pipeline...")
        class_name, confidence = predict_class(dummy_image_path)
        recyclable = is_recyclable(class_name)

        heatmap = apply_grad_cam(dummy_image_path, class_name)
        overlayed_img = overlay_heatmap(dummy_image_path, heatmap)

        # Save the overlayed image
        overlayed_img_path = "overlayed_image.jpg"
        cv2.imwrite(overlayed_img_path, overlayed_img)
        print(f"Overlayed image saved at {overlayed_img_path}")

        print("Pipeline execution complete.")
        print(f"Predicted Class: {class_name}, Confidence: {confidence}")
        print(f"Is Recyclable: {recyclable}")
    except Exception as e:
        print(f"Error in prediction pipeline: {e}")
