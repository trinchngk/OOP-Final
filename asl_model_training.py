import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

class ASLDataCollector:
    def __init__(self, output_dir="asl_dataset"):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # ASL alphabet labels (excluding J and Z which require motion)
        self.asl_labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
        ]
        
        for label in self.asl_labels:
            os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    def process_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
        return frame, results

    def save_hand_data(self, frame, results, label):
        if results.multi_hand_landmarks:
            # Get hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{label}_{timestamp}"
            
            # Save image
            image_path = os.path.join(self.output_dir, label, f"{filename}.jpg")
            cv2.imwrite(image_path, frame)
            
            # Save landmarks as numpy array
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmarks_path = os.path.join(self.output_dir, label, f"{filename}.npy")
            np.save(landmarks_path, landmarks_array)
            
            return True
        return False

    def run_capture(self):
        cap = cv2.VideoCapture(0)
        current_label_idx = 0
        samples_per_label = 0
        max_samples = 150  # Number of samples to collect per letter
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for selfie view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame, results = self.process_frame(frame)
            
            # Display current letter and instructions
            current_label = self.asl_labels[current_label_idx]
            cv2.putText(frame, f"Show letter: {current_label}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {samples_per_label}/{max_samples}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('ASL Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Press 's' to save sample
                if self.save_hand_data(frame, results, current_label):
                    samples_per_label += 1
                    
                    # Move to next letter if we have enough samples
                    if samples_per_label >= max_samples:
                        current_label_idx += 1
                        samples_per_label = 0
                        
                        # End if we've collected all letters
                        if current_label_idx >= len(self.asl_labels):
                            break
            
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    collector = ASLDataCollector()
    collector.run_capture()
