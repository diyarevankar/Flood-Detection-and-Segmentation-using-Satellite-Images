# app.py

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS # Needed to allow your frontend to talk to your backend
import torch
import numpy as np
import cv2
import os
import io
import base64
import tifffile as tiff
import segmentation_models_pytorch as smp

# --- Configuration ---
MODEL_WEIGHTS_PATH = 'model_weights/unet_custom_epoch_42.pth'
# Make sure to set this to your latest best model, e.g., 'unet_custom_epoch_142.pth'
# if that's truly your best, or stick to 'best_unet_custom_model.pth' if your training script
# consistently updates that file.

SAR_IMAGE_SIZE = (512, 512) # Or whatever size your model expects
FLOOD_THRESHOLD_PERCENT = 1.0 # Percentage of image pixels predicted as flood for "Flooded" status

app = Flask(__name__)

CORS(app) # This enables Cross-Origin Resource Sharing, essential for frontend-backend communication

# --- Model Loading (Global so it loads once when the server starts) ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Loads the trained PyTorch U-Net model."""
    global model
    try:
        # Re-define your model architecture as it was during training
        # IMPORTANT: These parameters must exactly match how you defined your model in train.py
        ARCH = 'unet'
        ENCODER = 'resnet34'
        ENCODER_WEIGHTS = None # Or 'imagenet' if you used pretrained weights
        ACTIVATION = 'sigmoid' # For binary segmentation

        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=2, # Assuming VV and VH channels
            classes=1,
            activation=ACTIVATION
        )
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        model.to(device)
        model.eval() # Set to evaluation mode
        print(f"Model loaded successfully from {MODEL_WEIGHTS_PATH} on {device}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}. Please check the path.")
        model = None # Indicate failure to load
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        model = None

# Load model when the app starts
with app.app_context():
    load_model()

# --- Image Preprocessing and Prediction Logic (Matching your training and single_image_prediction) ---

def preprocess_image(image_bytes):
    """
    Reads image bytes, preprocesses for model input.
    THIS MUST MATCH YOUR TRAINING PREPROCESSING EXACTLY.
    """
    # Assuming the input is a TIFF byte stream (from frontend upload)
    try:
        sar_image = tiff.imread(io.BytesIO(image_bytes))

        # Ensure 2 channels (C, H, W) and float32, as your model expects
        if sar_image.ndim == 2: # If grayscale (H, W), duplicate to 2 channels
            sar_image = np.stack([sar_image, sar_image], axis=0)
        elif sar_image.ndim == 3 and sar_image.shape[2] == 2: # If (H, W, C), permute to (C, H, W)
            sar_image = np.transpose(sar_image, (2, 0, 1))
        elif sar_image.ndim == 3 and sar_image.shape[0] == 2: # Already (C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected SAR image shape: {sar_image.shape}. Expected (H, W) or (H, W, 2) or (2, H, W).")

        # Convert to float32
        sar_image = sar_image.astype(np.float32)

        # Normalize if you applied specific normalization during training (e.g., min-max or mean/std)
        # For Sen1Floods11, simple min-max scaling to 0-1 might be appropriate if not done via custom transforms
        # If your training data ranges from -X to +Y, normalize accordingly.
        # Example: sar_image = (sar_image - sar_image.min()) / (sar_image.max() - sar_image.min()) # Simple min-max

        # Add batch dimension
        sar_image_tensor = torch.from_numpy(sar_image).unsqueeze(0)
        return sar_image_tensor

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise

def get_prediction_results(input_tensor):
    """Runs inference and generates output images and metrics."""
    if model is None:
        raise RuntimeError("Model not loaded. Cannot perform prediction.")

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs)
        predicted_mask_tensor = (probabilities > 0.5).float()

    # Convert tensors to numpy for visualization
    predicted_mask_np = predicted_mask_tensor.cpu().squeeze().numpy().astype(np.uint8)

    # For original image display, assume input_tensor[0,0,:,:] (first channel) is representative
    # Normalize original image for display (0-255)
    original_sar_display = input_tensor.cpu().squeeze().numpy()
    if original_sar_display.ndim == 3: # If 2 channels, take first for display
        original_sar_display = original_sar_display[0, :, :]
    
    original_sar_display = cv2.normalize(original_sar_display, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    original_sar_color = cv2.cvtColor(original_sar_display, cv2.COLOR_GRAY2BGR) # Convert to BGR for overlay

    # Create the red overlay
    red_overlay = np.zeros_like(original_sar_color)
    red_overlay[:, :, 2] = 255 # BGR format: Red channel is index 2

    # Apply mask for overlay
    masked_region = cv2.bitwise_and(red_overlay, red_overlay, mask=predicted_mask_np)
    combined_image_overlay = cv2.addWeighted(original_sar_color, 1, masked_region, 0.5, 0) # 0.5 opacity

    # Calculate percentage of flooded pixels
    flood_percentage = (np.sum(predicted_mask_np) / predicted_mask_np.size) * 100

    # Convert numpy arrays to base64 encoded JPEG for sending to frontend
    _, original_buffer = cv2.imencode('.jpg', original_sar_color)
    original_b64 = base64.b64encode(original_buffer).decode('utf-8')

    _, binary_buffer = cv2.imencode('.jpg', predicted_mask_np * 255) # Scale mask to 0/255 for JPEG
    binary_b64 = base64.b64encode(binary_buffer).decode('utf-8')

    _, overlay_buffer = cv2.imencode('.jpg', combined_image_overlay)
    overlay_b64 = base64.b64encode(overlay_buffer).decode('utf-8')

    return {
        "original_image": original_b64,
        "binary_mask": binary_b64,
        "overlay_image": overlay_b64,
        "flooded_percentage": flood_percentage,
        "overall_status": "Flooded" if flood_percentage > FLOOD_THRESHOLD_PERCENT else "Not Flooded"
    }

# --- Flask Routes ---

@app.route('/')
def home():
    return "Flood Detection Backend is running! Send POST requests to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    if 'sar_image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['sar_image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file:
        return jsonify({"error": "File is empty"}), 400

    try:
        image_bytes = file.read()
        input_tensor = preprocess_image(image_bytes)
        results = get_prediction_results(input_tensor)
        return jsonify(results)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # You can change the port if 5000 is in use
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=True restarts server on code changes