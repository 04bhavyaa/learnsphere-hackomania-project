# routes/proctoring_routes.py

from flask import Blueprint, Response, jsonify
import cv2
import mediapipe as mp
import time
import numpy as np
import threading

proctoring_bp = Blueprint('proctoring', __name__)

# Global variables to store proctoring data
proctoring_data = {
    'attention_status': 'UNKNOWN',
    'attention_percentage': 0,
    'num_faces': 0,
    'warnings': 0,
    'is_active': False
}

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define eye landmarks - using MediaPipe's specific iris landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]  # Main left eye landmarks
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  # Main right eye landmarks
LEFT_IRIS = [474, 475, 476, 477]  # Left iris landmarks
RIGHT_IRIS = [469, 470, 471, 472]  # Right iris landmarks

# Tracking variables
attention_start_time = None
inattention_start_time = None
attention_threshold = 3  # seconds
cheating_warnings = 0
last_warning_time = 0
warning_cooldown = 3  # seconds
total_attention_time = 0
total_inattention_time = 0

# Video capture
cap = None
face_mesh = None
proctoring_thread = None

def get_iris_position(iris_landmarks, eye_landmarks, frame):
    """Calculate iris position relative to eye corners"""
    # Get frame dimensions
    h, w, _ = frame.shape

    # Get iris center
    iris_x = sum([landmark.x for landmark in iris_landmarks]) / len(iris_landmarks)
    iris_y = sum([landmark.y for landmark in iris_landmarks]) / len(iris_landmarks)

    # Get eye corners - leftmost and rightmost points
    eye_points = [(landmark.x, landmark.y) for landmark in eye_landmarks]
    eye_left = min(eye_points, key=lambda p: p[0])[0]
    eye_right = max(eye_points, key=lambda p: p[0])[0]

    # Calculate relative position
    iris_pos = (iris_x - eye_left) / (eye_right - eye_left) if eye_right > eye_left else 0.5

    # Debug visualization - draw iris center and eye corners on the frame
    iris_px = int(iris_x * w)
    iris_py = int(iris_y * h)
    left_px = int(eye_left * w)
    right_px = int(eye_right * w)
    eye_y = int(iris_y * h)  # Use iris y for horizontal line

    # Draw points and line
    cv2.circle(frame, (iris_px, iris_py), 3, (0, 255, 0), -1)  # Iris center
    cv2.circle(frame, (left_px, eye_y), 3, (0, 0, 255), -1)  # Left corner
    cv2.circle(frame, (right_px, eye_y), 3, (0, 0, 255), -1)  # Right corner
    cv2.line(frame, (left_px, eye_y), (right_px, eye_y), (255, 0, 0), 1)

    # Determine position
    if iris_pos < 0.4:
        return "LEFT", iris_pos
    elif iris_pos > 0.65:
        return "RIGHT", iris_pos
    else:
        return "CENTER", iris_pos

def generate_frames():
    global cap, face_mesh, attention_start_time
    global total_attention_time, total_inattention_time, inattention_start_time
    global cheating_warnings, last_warning_time
    
    # Reset timing variables
    attention_start_time = time.time()
    inattention_start_time = None
    total_attention_time = 0
    total_inattention_time = 0
    cheating_warnings = 0
    last_warning_time = 0
    
    while proctoring_data['is_active']:
        success, frame = cap.read()
        if not success:
            continue

        # Flip the frame horizontally for a more natural view (mirror)
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Process the frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        # Variables for current frame
        num_faces = 0
        left_eye_pos = "UNKNOWN"
        right_eye_pos = "UNKNOWN"
        eye_positions = []  # Store positions for averaging

        # Draw face landmarks and analyze eye positions
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)

            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Focus on the first face for eye tracking
                if face_idx == 0:
                    # Get specific eye landmarks
                    left_eye_landmarks = [face_landmarks.landmark[i] for i in LEFT_EYE]
                    right_eye_landmarks = [face_landmarks.landmark[i] for i in RIGHT_EYE]
                    left_iris_landmarks = [face_landmarks.landmark[i] for i in LEFT_IRIS]
                    right_iris_landmarks = [face_landmarks.landmark[i] for i in RIGHT_IRIS]

                    # Determine iris positions
                    left_pos, left_ratio = get_iris_position(left_iris_landmarks, left_eye_landmarks, frame)
                    right_pos, right_ratio = get_iris_position(right_iris_landmarks, right_eye_landmarks, frame)

                    left_eye_pos = left_pos
                    right_eye_pos = right_pos
                    eye_positions.append(left_ratio)
                    eye_positions.append(right_ratio)

                    # Draw eye landmarks
                    for landmark_idx, landmark in enumerate(face_landmarks.landmark):
                        landmark_px = int(landmark.x * w)
                        landmark_py = int(landmark.y * h)

                        # Draw only eyes and iris landmarks to reduce clutter
                        if landmark_idx in LEFT_EYE + RIGHT_EYE + LEFT_IRIS + RIGHT_IRIS:
                            color = (0, 255, 0) if landmark_idx in LEFT_IRIS + RIGHT_IRIS else (255, 0, 0)
                            cv2.circle(frame, (landmark_px, landmark_py), 1, color, -1)

                # For all faces, draw bounding box
                face_points = [(landmark.x * w, landmark.y * h) for landmark in face_landmarks.landmark]
                face_left = int(min([p[0] for p in face_points]))
                face_right = int(max([p[0] for p in face_points]))
                face_top = int(min([p[1] for p in face_points]))
                face_bottom = int(max([p[1] for p in face_points]))

                # Draw face bounding box
                box_color = (0, 255, 0) if face_idx == 0 else (0, 0, 255)
                cv2.rectangle(frame, (face_left, face_top), (face_right, face_bottom), box_color, 1)

        # Determine attention status based on eye positions
        attention_status = "UNKNOWN"
        if eye_positions:
            # Average eye position
            avg_position = sum(eye_positions) / len(eye_positions)

            # Check if eyes are centered
            if 0.4 <= avg_position <= 0.65:
                attention_status = "ATTENTIVE"
                if inattention_start_time is not None:
                    total_inattention_time += time.time() - inattention_start_time
                    inattention_start_time = None
            else:
                attention_status = "DISTRACTED"
                if inattention_start_time is None:
                    inattention_start_time = time.time()

        # Attention tracking
        if attention_status == "ATTENTIVE" and num_faces == 1:
            # Reset inattention timer
            inattention_start_time = None

            # Calculate total attentive time
            total_attention_time = time.time() - attention_start_time - total_inattention_time
        else:
            # Track inattention
            if inattention_start_time is None:
                inattention_start_time = time.time()

            # Show warning if inattentive for too long
            if inattention_start_time and (time.time() - inattention_start_time) > attention_threshold:
                if time.time() - last_warning_time > warning_cooldown:
                    cv2.putText(frame, 'Please focus on the screen', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    last_warning_time = time.time()

        # Multiple faces check (cheating detection)
        if num_faces > 1:
            if time.time() - last_warning_time > warning_cooldown:
                cheating_warnings += 1
                cv2.putText(frame, f'Warning: {num_faces} faces detected', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                last_warning_time = time.time()

        # Calculate attention percentage
        total_time = time.time() - attention_start_time
        attention_percentage = (total_attention_time / total_time) * 100 if total_time > 0 else 0

        # Update global proctoring data
        proctoring_data['attention_status'] = attention_status
        proctoring_data['attention_percentage'] = round(attention_percentage, 1)
        proctoring_data['num_faces'] = num_faces
        proctoring_data['warnings'] = cheating_warnings

        # Display eye positions and status information
        eye_position_text = f"Left Eye: {left_eye_pos}, Right Eye: {right_eye_pos}"
        cv2.putText(frame, eye_position_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display attention status
        status_color = (0, 255, 0) if attention_status == "ATTENTIVE" else (0, 0, 255)
        cv2.putText(frame, f"Status: {attention_status}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Display attention percentage
        cv2.putText(frame, f"Attention: {attention_percentage:.1f}%", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display number of faces
        cv2.putText(frame, f"Faces: {num_faces}", (w - 100, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Convert to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame for the stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def proctoring_background_task():
    global cap, face_mesh
    
    # Initialize Face Mesh with iris tracking enabled
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,  # Enable iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Access the webcam
    cap = cv2.VideoCapture(0)
    
    # Generate frames in a loop
    for frame in generate_frames():
        pass
    
    # Clean up resources
    if cap:
        cap.release()
    if face_mesh:
        face_mesh.close()

@proctoring_bp.route('/start', methods=['GET','POST'])
def start_proctoring():
    global proctoring_thread
    
    if not proctoring_data['is_active']:
        proctoring_data['is_active'] = True
        proctoring_thread = threading.Thread(target=proctoring_background_task)
        proctoring_thread.daemon = True
        proctoring_thread.start()
        return jsonify({"status": "Proctoring started"})
    else:
        return jsonify({"status": "Proctoring already active"})

@proctoring_bp.route('/stop', methods=['POST'])
def stop_proctoring():
    global proctoring_thread
    
    if proctoring_data['is_active']:
        proctoring_data['is_active'] = False
        if proctoring_thread:
            proctoring_thread.join(timeout=1.0)
        return jsonify({"status": "Proctoring stopped"})
    else:
        return jsonify({"status": "Proctoring already inactive"})

@proctoring_bp.route('/status', methods=['GET'])
def get_proctoring_status():
    return jsonify(proctoring_data)

@proctoring_bp.route('/video_feed')
def video_feed():
    if proctoring_data['is_active']:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Proctoring is not active", 400