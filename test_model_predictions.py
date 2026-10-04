#!/usr/bin/env python3
"""
Test script to verify that models are making actual predictions instead of random ones
"""

import pickle
import numpy as np
import os

def test_actual_model_predictions():
    """Test that models are actually making predictions, not random ones"""
    print("Testing Actual Model Predictions")
    print("=" * 50)
    
    # Load models
    models_dir = "new_models"
    
    try:
        # Load vectorizer
        with open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded")
        
        # Load models
        model_files = {
            'Naive Bayes': "naive_bayes.pkl",
            'SVM': "svm.pkl", 
            'BPNN': "bpnn.pkl"
        }
        
        models = {}
        for model_name, filename in model_files.items():
            with open(os.path.join(models_dir, filename), 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f"✅ {model_name} loaded")
        
        # Test text - should be clearly fake news
        test_text = "A widely shared article claims that extraterrestrial beings have secretly landed on Earth and are now living undercover in major cities, working alongside government officials."
        
        # Preprocess text (same as in app)
        processed_text = test_text.lower()
        import re
        processed_text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|<.*?>', '', processed_text)
        processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
        processed_text = ' '.join(processed_text.split())
        words = processed_text.split()
        words = [word for word in words if len(word) >= 2]
        processed_text = ' '.join(words)
        
        print(f"\nTest text: {test_text[:50]}...")
        print(f"Processed: {processed_text[:50]}...")
        
        # Vectorize
        text_tfidf = vectorizer.transform([processed_text])
        print(f"Vectorized shape: {text_tfidf.shape}")
        
        # Test each model
        print(f"\nModel Predictions:")
        print("-" * 30)
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Get probabilities
                    probabilities = model.predict_proba(text_tfidf)[0]
                    fake_prob = probabilities[0]  # Class 0 = Fake
                    real_prob = probabilities[1]  # Class 1 = Real
                    
                    # Make prediction
                    prediction = model.predict(text_tfidf)[0]
                    result = "REAL" if prediction == 1 else "FAKE"
                    
                    print(f"{model_name:12} | {result:4} | Real: {real_prob:.3f} | Fake: {fake_prob:.3f}")
                    
                else:
                    # For models without predict_proba
                    prediction = model.predict(text_tfidf)[0]
                    result = "REAL" if prediction == 1 else "FAKE"
                    print(f"{model_name:12} | {result:4} | (No probabilities available)")
                    
            except Exception as e:
                print(f"{model_name:12} | ERROR: {e}")
        
        # Test consistency - run same prediction multiple times
        print(f"\nConsistency Test (same text, multiple runs):")
        print("-" * 45)
        
        for i in range(3):
            nb_pred = models['Naive Bayes'].predict(text_tfidf)[0]
            nb_result = "REAL" if nb_pred == 1 else "FAKE"
            print(f"Run {i+1}: Naive Bayes = {nb_result}")
        
        print(f"\n✅ Model testing complete!")
        print(f"If all runs show the same result, models are working correctly.")
        print(f"If results vary randomly, there's still an issue.")
        
    except Exception as e:
        print(f"❌ Error testing models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actual_model_predictions()
