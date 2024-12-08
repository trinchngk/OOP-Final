import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, models
import glob

class ASLModelTrainer:
    def __init__(self, dataset_path="asl_dataset"):
        self.dataset_path = dataset_path
        self.labels = sorted([d for d in os.listdir(dataset_path) 
                            if os.path.isdir(os.path.join(dataset_path, d))])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
    def load_data(self):
        X = []
        y = []
        
        # Load all .npy files for each label
        for label in self.labels:
            label_path = os.path.join(self.dataset_path, label)
            landmark_files = glob.glob(os.path.join(label_path, "*.npy"))
            
            for landmark_file in landmark_files:
                # Load landmark data
                landmarks = np.load(landmark_file)
                # Flatten the 21x3 landmarks into a 63-dimensional vector
                landmarks_flat = landmarks.flatten()
                
                X.append(landmarks_flat)
                y.append(self.label_to_idx[label])
        
        return np.array(X), np.array(y)
    
    def create_model(self):
        model = models.Sequential([
            layers.Input(shape=(63,)),  # 21 landmarks × 3 coordinates
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self):
        # Load and preprocess data
        X, y = self.load_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train the model
        model = self.create_model()
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"\nTest accuracy: {test_accuracy:.4f}")
        
        # Save the model in H5 format
        model.save("models/asl_recognition_model.keras")
        
        return model, history

if __name__ == "__main__":
    trainer = ASLModelTrainer()
    model, history = trainer.train()