import sys
import pickle
import numpy

# Add internal numpy functions that might be used during unpickling
if not hasattr(numpy, '_reconstruct'):
    numpy._reconstruct = lambda array, shape, dtype: numpy.ndarray(shape, dtype)

# Create a custom unpickler that handles numpy compatibility issues
class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle numpy special cases
        if module == 'numpy.core.multiarray':
            if name == '_reconstruct':
                return numpy._reconstruct
            else:
                module = 'numpy'
        elif module == "numpy._core.multiarray":
            module = "numpy"
            name = name.replace("_multiarray", "")
        # Remap other numpy modules as needed
        elif module.startswith("numpy._"):
            module = "numpy"
        elif module.startswith("numpy.core"):
            module = "numpy"
            
        try:
            return super().find_class(module, name)
        except AttributeError as e:
            if "numpy" in module and hasattr(numpy, name):
                return getattr(numpy, name)
            raise e

def safe_load_pickle(file_path):
    """Load a pickle file with numpy compatibility fixes"""
    try:
        with open(file_path, "rb") as f:
            return NumpyCompatUnpickler(f).load()
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        # Try a fallback approach for TF-IDF vectorizer
        if "tfidf_vectorizer" in file_path:
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("Creating a new TfidfVectorizer as a fallback")
            return TfidfVectorizer()
        elif "naive_bayes" in file_path:
            from sklearn.naive_bayes import MultinomialNB
            print("Creating a new MultinomialNB as a fallback")
            return MultinomialNB()
        elif "svm" in file_path:
            from sklearn.svm import SVC
            print("Creating a new SVC as a fallback")
            return SVC(probability=True)
        elif "bpnn" in file_path:
            from sklearn.neural_network import MLPClassifier
            print("Creating a new MLPClassifier as a fallback")
            return MLPClassifier()
        elif "scaler" in file_path:
            from sklearn.preprocessing import StandardScaler
            print("Creating a new StandardScaler as a fallback")
            return StandardScaler()
        elif "preprocessing" in file_path:
            # Define a basic preprocessing function
            def preprocess_text(text):
                import re
                if not text:
                    return ""
                # Convert to lowercase
                text = text.lower()
                # Remove URLs, email addresses, HTML tags
                text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|<.*?>', '', text)
                # Remove punctuation and special characters
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                # Remove extra whitespaces
                text = ' '.join(text.split())
                return text
            print("Creating a basic preprocessing function as a fallback")
            return preprocess_text
        else:
            raise e
