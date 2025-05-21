import cv2
from deepface import DeepFace
import time
import numpy as np

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)
    
    # Convert back to BGR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def smooth_emotions(emotion_buffer, current_emotions):
    if not emotion_buffer:
        return current_emotions
    
    # Calculate weighted average of emotions
    smoothed_emotions = []
    weights = np.linspace(0.5, 1.0, len(emotion_buffer) + 1)  # More weight to recent emotions
    
    for face_idx in range(len(current_emotions)):
        avg_emotions = {}
        total_weight = 0
        
        # Add current emotions with highest weight
        current_face = current_emotions[face_idx]
        current_weight = weights[-1]
        emotions = current_face.get('emotion', {})
        for emotion_name, score in emotions.items():
            avg_emotions[emotion_name] = score * current_weight
        total_weight = current_weight
        
        # Add historical emotions with decreasing weights
        for i, buffer_item in enumerate(emotion_buffer):
            if face_idx < len(buffer_item):
                face_data = buffer_item[face_idx]
                emotions = face_data.get('emotion', {})
                weight = weights[i]
                for emotion_name, score in emotions.items():
                    avg_emotions[emotion_name] = avg_emotions.get(emotion_name, 0) + score * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            for emotion_name in avg_emotions:
                avg_emotions[emotion_name] /= total_weight
            
            # Create smoothed face data
            smoothed_face = current_face.copy()
            smoothed_face['emotion'] = avg_emotions
            smoothed_face['dominant_emotion'] = max(avg_emotions.items(), key=lambda x: x[1])[0]
            smoothed_emotions.append(smoothed_face)
    
    return smoothed_emotions

def main():
    # Initialize the webcam with specific settings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting facial expression recognition... Press 'q' to quit")
    print("Please wait while the models are loading...")
    
    emotion_colors = {
        'angry': (0, 0, 255),    # Red
        'disgust': (0, 140, 255),  # Orange
        'fear': (0, 255, 255),   # Yellow
        'happy': (0, 255, 0),    # Green
        'sad': (255, 0, 0),      # Blue
        'surprise': (255, 0, 255),# Purple
        'neutral': (128, 128, 128)# Gray
    }
    
    try:
        DeepFace.build_model("Emotion")
        print("Model loaded! Starting emotion detection...")
    except Exception as e:
        print(f"Error loading model: {e}")
        cap.release()
        return
    
    last_process_time = time.time()
    process_every_n_seconds = 0.2  # Process more frequently
    
    prev_emotions = None
    emotion_buffer = []
    buffer_size = 7  # Increased buffer size
    
    # Adjusted confidence thresholds
    min_confidence = 12  # Lower threshold for initial detection
    display_confidence = 15  # Higher threshold for display
    
    while True:
        try:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Error: Failed to capture frame")
                continue
            
            # Apply preprocessing
            processed_frame = preprocess_frame(frame)
            
            # Convert to RGB for DeepFace
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            current_time = time.time()
            should_process_emotions = current_time - last_process_time >= process_every_n_seconds
            
            if should_process_emotions:
                try:
                    # Analyze emotions with improved settings
                    demography = DeepFace.analyze(processed_frame, 
                                                actions=['emotion'],
                                                detector_backend='mtcnn',
                                                enforce_detection=False,
                                                silent=True,
                                                align=True)
                    print(f"DeepFace.analyze returned: {demography}")
                except Exception as e:
                    print(f"DeepFace.analyze error: {e}")
                    continue
                # Handle tuple output (DeepFace may return (result, region) tuple)
                if isinstance(demography, tuple):
                    demography = demography[0]
                # If dict, wrap in list
                if isinstance(demography, dict):
                    demography = [demography]
                
                if isinstance(demography, list) and len(demography) > 0:
                    # Update emotion buffer
                    emotion_buffer.append(demography)
                    if len(emotion_buffer) > buffer_size:
                        emotion_buffer.pop(0)
                    
                    # Apply sophisticated smoothing
                    prev_emotions = smooth_emotions(emotion_buffer, demography)
                    last_process_time = current_time
            
            # Display results
            if prev_emotions:
                for face_data in prev_emotions:
                    try:
                        region = face_data.get('region', {})
                        x = region.get('x', 0)
                        y = region.get('y', 0)
                        w = region.get('w', 0)
                        h = region.get('h', 0)
                        
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        
                        emotions = face_data.get('emotion', {})
                        dominant_emotion = face_data.get('dominant_emotion', 'unknown')
                        emotion_color = emotion_colors.get(dominant_emotion, (255, 255, 255))
                        
                        # Display dominant emotion with higher confidence threshold
                        confidence = emotions.get(dominant_emotion, 0)
                        if confidence >= display_confidence:
                            text = f"{dominant_emotion} ({confidence:.1f}%)"
                            cv2.putText(frame, text, (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)
                            
                            # Display all emotions above minimum threshold
                            y_offset = y + h + 20
                            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                            for emotion_name, score in sorted_emotions:
                                if score > min_confidence:
                                    text = f"{emotion_name}: {score:.1f}%"
                                    cv2.putText(frame, text, (x, y_offset),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                              emotion_colors[emotion_name], 1)
                                    y_offset += 20
                    
                    except Exception as e:
                        print(f"Error displaying results: {e}")
            
            cv2.imshow('Facial Expression Recognition', frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 