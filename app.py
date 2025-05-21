from flask import Flask, render_template, request, jsonify
import cv2
from deepface import DeepFace
import numpy as np
import base64
import io
from PIL import Image
import traceback

app = Flask(__name__)

def preprocess_frame(frame):
    """
    Enhance the image quality for better face detection
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)
    
    # Convert back to BGR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image was provided
        if 'image' not in request.files:
            return jsonify({'error': 'Please capture an image first'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image was captured'}), 400
        
        # Read and validate the image
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({'error': 'The captured image is empty'}), 400
            
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Unable to process the image. Please try again'}), 400
        
        # Enhance image quality
        processed_frame = preprocess_frame(frame)
        
        # Analyze emotions
        try:
            result = DeepFace.analyze(processed_frame, 
                                    actions=['emotion'],
                                    detector_backend='mtcnn',
                                    enforce_detection=False,
                                    silent=True,
                                    align=True)
            
            # Handle different result formats
            if isinstance(result, list):
                if len(result) == 0:
                    return jsonify({'error': 'No face detected. Please make sure your face is clearly visible in the frame'}), 400
                result = result[0]  # Take the first face detected
            elif isinstance(result, tuple):
                result = result[0]  # Take the first element of the tuple
            
            # Ensure we have emotion data
            if 'emotion' not in result:
                return jsonify({'error': 'Unable to detect emotions. Please try again with better lighting'}), 400
            
            # Get the dominant emotion
            emotions = result['emotion']
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions[dominant_emotion]
            
            return jsonify({
                'emotion': dominant_emotion,
                'confidence': confidence,
                'all_emotions': emotions
            })
            
        except Exception as e:
            error_message = str(e)
            if "No face detected" in error_message:
                return jsonify({'error': 'No face detected. Please make sure your face is clearly visible in the frame'}), 400
            elif "enforce_detection" in error_message:
                return jsonify({'error': 'Face detection failed. Please try again with better lighting'}), 400
            else:
                return jsonify({'error': 'Unable to analyze emotions. Please try again'}), 400
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")  # This will print to the server console
        return jsonify({
            'error': 'An unexpected error occurred. Please try again'
        }), 500

if __name__ == '__main__':
    # Initialize the model
    try:
        DeepFace.build_model("Emotion")
        print("Emotion detection model loaded successfully!")
    except Exception as e:
        print(f"Error loading emotion detection model: {e}")
    
    app.run(debug=True) 