"""
This script attempts to load the vectorizer model with multiple methods
to determine which approach works best with the current numpy version.
"""

import os
import pickle
import sys
import numpy
import traceback

print(f"Numpy version: {numpy.__version__}")

def standard_load(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        print("Standard loading: SUCCESS")
        return True
    except Exception as e:
        print(f"Standard loading error: {e}")
        traceback.print_exc()
        return False

def try_register_numpy_core(file_path):
    try:
        # Add compatibility for numpy version issues
        sys.modules['numpy._core'] = numpy
        if not hasattr(numpy, '_core'):
            numpy._core = numpy
        if not hasattr(numpy, '_core.multiarray'):
            numpy._core.multiarray = numpy
            
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        print("Loading with numpy._core mapping: SUCCESS")
        return True
    except Exception as e:
        print(f"numpy._core mapping error: {e}")
        traceback.print_exc()
        return False

def try_custom_unpickler(file_path):
    # Custom unpickler class
    class NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Handle numpy special cases
            if module == 'numpy.core.multiarray' or module == 'numpy._core.multiarray':
                module = 'numpy'
            elif module.startswith("numpy._"):
                module = "numpy"
            elif module.startswith("numpy.core"):
                module = "numpy"
            return super().find_class(module, name)
    
    try:
        with open(file_path, 'rb') as file:
            obj = NumpyCompatUnpickler(file).load()
        print("Custom unpickler: SUCCESS")
        return True
    except Exception as e:
        print(f"Custom unpickler error: {e}")
        traceback.print_exc()
        return False

def main():
    models_dir = "models"
    model_file = None
    
    # Find the vectorizer file
    for file in os.listdir(models_dir):
        if "vectorizer" in file:
            model_file = os.path.join(models_dir, file)
            break
    
    if not model_file:
        print("No vectorizer file found!")
        return
    
    print(f"Testing loading of file: {model_file}")
    
    # Try different methods
    print("\n1. TRYING STANDARD PICKLE LOADING:")
    standard_result = standard_load(model_file)
    
    print("\n2. TRYING WITH NUMPY._CORE MAPPING:")
    core_result = try_register_numpy_core(model_file)
    
    print("\n3. TRYING WITH CUSTOM UNPICKLER:")
    unpickler_result = try_custom_unpickler(model_file)
    
    # Report results
    print("\nRESULTS SUMMARY:")
    print(f"Standard loading: {'SUCCESS' if standard_result else 'FAILED'}")
    print(f"Numpy._core mapping: {'SUCCESS' if core_result else 'FAILED'}")
    print(f"Custom unpickler: {'SUCCESS' if unpickler_result else 'FAILED'}")

if __name__ == "__main__":
    main()
