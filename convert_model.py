import tensorflow as tf
import tensorflowjs as tfjs
import os

def convert_model(input_path, output_dir):
    
    # Load the Keras model
    model = tf.keras.models.load_model(input_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert and save the model
    tfjs.converters.save_keras_model(model, output_dir)
    print(f"Model converted and saved to {output_dir}")

if __name__ == "__main__":
    # Paths for model conversion
    input_model_path = "models/asl_recognition_model.h5"
    output_model_dir = "models/tfjs_model"
    
    convert_model(input_model_path, output_model_dir)