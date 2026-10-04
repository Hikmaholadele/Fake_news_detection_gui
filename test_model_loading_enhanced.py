"""
Enhanced test script to load models using the custom unpickler for numpy compatibility
"""

import os
import pickle
import sys
import numpy
from numpy_compat import safe_load_pickle

def test_model_loading():
    print("Testing model loading...")
    models_dir = "models"
    
    try:
        # Check if models directory exists
        if not os.path.exists(models_dir):
            print(f"Error: Directory '{models_dir}' does not exist!")
            return False
        
        # List all files in the directory
        files = os.listdir(models_dir)
        print(f"Files in {models_dir}: {files}")
        
        # Look for the all_components file
        all_components_files = [f for f in files if f.endswith('all_components.pkl')]
        
        if not all_components_files:
            print("Error: No all_components.pkl file found!")
            return False
        
        # Sort by name (which includes timestamp) to get the most recent
        all_components_files.sort(reverse=True)
        all_components_path = os.path.join(models_dir, all_components_files[0])
        print(f"Found all components file: {all_components_path}")
        
        # Try loading the file using our safe loader
        print(f"Attempting to load components from {all_components_path}...")
        components = safe_load_pickle(all_components_path)
        
        # Check what's inside
        print("Successfully loaded components!")
        print(f"Keys in components: {list(components.keys())}")
        
        models = components.get('models', {})
        print(f"Models: {list(models.keys())}")
        
        vectorizer = components.get('vectorizer')
        print(f"Vectorizer available: {vectorizer is not None}")
        
        scaler = components.get('scaler')
        print(f"Scaler available: {scaler is not None}")
        
        preprocess_text = components.get('preprocess_text')
        print(f"Preprocessing function available: {preprocess_text is not None}")
        
        # Test preprocessing a sample text
        if preprocess_text is not None:
            sample_text = "This is a test article about fake news."
            try:
                processed = preprocess_text(sample_text)
                print(f"Preprocessing test: SUCCESS")
                print(f"Sample text: '{sample_text}'")
                print(f"Processed: '{processed}'")
            except Exception as e:
                print(f"Preprocessing test failed: {str(e)}")
        
        # Test vectorizing a sample text
        if vectorizer is not None and preprocess_text is not None:
            try:
                sample_text = "This is a test article about fake news."
                processed = preprocess_text(sample_text)
                vectorized = vectorizer.transform([processed])
                print(f"Vectorization test: SUCCESS")
                print(f"Vectorized shape: {vectorized.shape}")
            except Exception as e:
                print(f"Vectorization test failed: {str(e)}")
        
        print("All tests completed!")
        return True
        
    except Exception as e:
        print(f"Error during model loading test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()
