from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from waitress import serve
from flask_cors import CORS
import logging
import os
import cv2
import base64
import numpy as np
import tensorflow as tf
import mediapipe as mp

class ASLRecognizerServer:

    def __init__(self, model_path="models/asl_recognition_model.keras"):

        # log for debugging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.app = Flask(__name__, static_folder='templates', template_folder='templates') 
        CORS(self.app)

        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        self.logger.info(f"loaded {model_path}")
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # ASL labels (in the same order as training)
        self.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
        ]

        # routes
        @self.app.route('/')
        def index():
            return render_template('index.html')
        @self.app.route('/model/<path:path>')
        def model(path):
            return send_from_directory('models/tfjs_model', path)
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('Client connected')
        @self.socketio.on('predict')
        def handle_predict(data):
            prediction = self.process_frame(data['frame'])
            if prediction:
                emit('prediction', prediction)

    def process_frame(self, image_data):
        # try:
            # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        prediction = None
        confidence = 0
        
        # If hands are detected, make a prediction
        if results.multi_hand_landmarks:
            # Get landmarks and make prediction
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
            landmarks_flat = landmarks.flatten()
            
            # Reshape input to match training data
            landmarks_flat = landmarks_flat.reshape(1, -1)
            
            # Make prediction
            # try:
            pred = self.model.predict(landmarks_flat, verbose=0)
            predicted_idx = np.argmax(pred[0])
            confidence = float(pred[0][predicted_idx])
            prediction = self.labels[predicted_idx]
            # except Exception as pred_err:
            #     self.logger.error(f"Prediction error: {pred_err}")
            #     return None
        
        return {
            'letter': prediction,
            'confidence': confidence
        }
    # except Exception as e:
    #     self.logger.error(f"Error processing frame: {e}")
    #     return None
    
    

    def run(self, host='0.0.0.0', port=8000, debug=True):
        self.logger.info(f"Starting server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def index():
    return render_template('index.html')

def main():
    server = ASLRecognizerServer()
    server.run()

if __name__ == '__main__':
    main()