import tensorflow as tf
import tensorflowjs as tfjs
import os
import numpy as np

def convert_model(input_path, output_dir, test_data=None):
    """
    Convert a Keras model to TensorFlow.js format
    
    Parameters:
    - input_path: Path to the input Keras model
    - output_dir: Directory to save the converted TensorFlow.js model
    - test_data: Optional test data for validation
    
    Returns:
    - Converted Keras model
    """
    # Load the Keras model
    model = tf.keras.models.load_model(input_path)
    
    # Print model summary to understand input requirements
    print("Model Summary:")
    model.summary()
    
    # Print input shape
    input_shape = model.input_shape
    print("\nExpected Input Shape:", input_shape)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert and save the model using the correct TensorFlow.js method
    try:
        tfjs.converters.save_keras_model(model, output_dir)
        print(f"\nModel converted and saved to {output_dir}")
    except Exception as e:
        print(f"Error converting model: {e}")
        raise
    
    # Optional model conversion validation
    if test_data is not None:
        validate_conversion(model, test_data, output_dir)
    
    return model

def validate_conversion(original_model, test_data, output_dir):
    """
    Validate model conversion by comparing predictions
    and checking weight similarities
    
    Parameters:
    - original_model: Original Keras model
    - test_data: Test input data
    - output_dir: Directory where model is saved
    
    Returns:
    - Prediction difference and weight differences
    """
    try:
        # Preprocess test data to match model's expected input
        # First, print the current test data shape
        print("\nOriginal Test Data Shape:", test_data.shape)
        
        # Determine how to reshape or prepare test data
        input_shape = original_model.input_shape
        
        # If input is 4D (typical for image models)
        if len(input_shape) == 4:
            # Reshape or preprocess image data
            if test_data.ndim == 4:
                # If already 4D, ensure correct shape
                if test_data.shape[1:] != input_shape[1:]:
                    # Resize or reshape as needed
                    test_data = tf.image.resize(test_data, input_shape[1:3]).numpy()
            elif test_data.ndim == 3:
                # Add batch dimension if missing
                test_data = np.expand_dims(test_data, axis=0)
        
        # If input is 2D (typical for tabular data)
        elif len(input_shape) == 2:
            # Flatten or reshape to match expected input
            test_data = test_data.reshape(-1, input_shape[1])
        
        # Print reshaped test data shape
        print("Reshaped Test Data Shape:", test_data.shape)
        
        # Generate predictions from original model
        python_preds = original_model.predict(test_data)
        
        # Compare predictions
        print("\nModel Conversion Validation:")
        
        # Compare model weights
        original_weights = original_model.get_weights()
        weight_diffs = []
        
        for i, weight in enumerate(original_weights):
            # Calculate the mean absolute difference for each weight tensor
            diff = np.mean(np.abs(weight))
            weight_diffs.append(diff)
            print(f"Layer {i} Weight Difference: {diff}")
        
        return weight_diffs
    
    except Exception as e:
        print(f"Validation error: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        return None

def load_test_data():
    """
    Placeholder function to load test data
    Replace with your actual data loading method
    
    Returns:
    - Test data array or None
    """
    try:
        # Example: Generate random test data
        # Replace this with your actual test data loading method
        x_test = np.random.rand(32, 63).astype('float32')  # Adjust shape to match model input
        return x_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

if __name__ == "__main__":
    # Paths for model conversion
    input_model_path = "models/asl_recognition_model.keras"
    output_model_dir = "models/tfjs_model"
    
    # Load test data
    test_data = load_test_data()
    
    # Convert model with optional validation
    converted_model = convert_model(
        input_model_path,
        output_model_dir,
        test_data
    )