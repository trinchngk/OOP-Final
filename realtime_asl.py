import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

class ASLRecognizer:
    def __init__(self, model_path="asl_recognition_model.h5"):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
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
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        prediction = None
        confidence = 0
        
        # If hands are detected, make a prediction
        if results.multi_hand_landmarks:
            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
            
            # Get landmarks and make prediction
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
            landmarks_flat = landmarks.flatten()
            
            # Make prediction
            pred = self.model.predict(np.array([landmarks_flat]), verbose=0)
            predicted_idx = np.argmax(pred[0])
            confidence = pred[0][predicted_idx]
            prediction = self.labels[predicted_idx]
        
        return frame, prediction, confidence
    
    def run(self):
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Process frame and get prediction
            frame, prediction, confidence = self.process_frame(frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display prediction if available
            if prediction:
                cv2.putText(frame, f"Letter: {prediction} ({confidence:.2f})", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('ASL Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    recognizer = ASLRecognizer()
    recognizer.run()