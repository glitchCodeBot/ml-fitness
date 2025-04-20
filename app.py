from flask import Flask, render_template, request
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from camera_processor import CameraProcessor
from exercise_analyzer import ExerciseAnalyzer
import eventlet
import threading

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Initialize components
camera_processor = CameraProcessor()
analyzer = ExerciseAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def process_image(image_data):
    # Convert base64 image to OpenCV format
    img_bytes = base64.b64decode(image_data.split(',')[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Process frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    angles = camera_processor.process_frame(frame_rgb)
    
    if angles:
        exercise = analyzer.classify_exercise(angles)
        feedback = analyzer.check_form(exercise, angles)
        
        # In a real app, you'd track reps over time
        rep_count = 0  # You'd implement proper rep counting
        
        return {
            'exercise': exercise,
            'repCount': rep_count,
            'feedback': feedback
        }
    return None

@socketio.on('message')
def handle_message(data):
    if 'image' in data:
        result = process_image(data['image'])
        if result:
            socketio.emit('update', result)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)