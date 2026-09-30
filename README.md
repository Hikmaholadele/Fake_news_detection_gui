#  Fake News Detection with GUI

An advanced machine learning system for detecting fake news using multiple ML models with a modern graphical user interface built with Python and Tkinter.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GUI](https://img.shields.io/badge/GUI-Tkinter-orange.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-red.svg)

## 🎯 Overview

This application provides an intuitive desktop interface for analyzing news articles and determining their authenticity using multiple machine learning algorithms. The system combines the power of three different ML models to provide robust and reliable fake news detection.

## Features

- **Multi-Model Analysis**: Uses Naive Bayes, SVM, and Neural Network models
- ** Modern GUI**: Clean, professional interface with real-time analysis
- ** Interactive Visualizations**: Charts and graphs showing model predictions
- ** Confidence Scoring**: Detailed confidence levels for each prediction
- ** Analysis History**: Track and review previous analyses
- ** Real-time Processing**: Fast text analysis with threading support
- ** Ensemble Voting**: Combines multiple models for enhanced accuracy

##  Machine Learning Models

### 1. **Naive Bayes Classifier**

- **Type**: Probabilistic classifier
- **Strengths**: Fast, simple, works well with small datasets
- **Use Case**: Baseline classification with strong independence assumptions

### 2. **Support Vector Machine (SVM)**

- **Type**: Linear/Non-linear classifier
- **Strengths**: Effective for text classification, handles high-dimensional data
- **Use Case**: Finding optimal hyperplane to separate fake and real news

### 3. **Backpropagation Neural Network (BPNN)**

- **Type**: Multi-layer perceptron
- **Strengths**: Can learn complex patterns, often provides best accuracy
- **Use Case**: Deep pattern recognition in text features

### 4. **Weighted Ensemble**

- **Type**: Meta-classifier
- **Strengths**: Combines strengths of all models
- **Use Case**: Final robust prediction using weighted voting

##  Technology Stack

- **GUI Framework**: Tkinter with modern styling
- **ML Libraries**: scikit-learn
- **Text Processing**: TF-IDF Vectorization, Regex preprocessing
- **Visualization**: matplotlib, seaborn
- **Data Handling**: pandas, numpy
- **Threading**: For non-blocking UI operations

##  Requirements

### System Requirements

- Python 3.8 or higher
- Windows/Linux/macOS
- Minimum 4GB RAM
- 500MB free disk space

### Python Dependencies

```
tkinter (usually comes with Python)
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

##  Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Hikmaholadele/Fake_news_detection_gui.git
cd fake_news_detection_gui
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

### 3. Run the Application

```bash
python fake_news_detector_app.py
```

##  Project Structure

```
fake_news_detection_with_gui/
│
├── fake_news_detector_app.py      # Main application file
├── models/                        # Pre-trained model files
│   ├── *.pkl                     # Pickled model components
│   └── ...
├── requirements.txt               # Python dependencies
├── README.md                     # Project documentation
└── assets/                       # UI assets (optional)
    └── app_icon.ico              # Application icon
```

##  How to Use

### 1. **Launch the Application**

Run the Python script to open the GUI interface.

### 2. **Enter News Text**

- Type or paste the news article text in the input area
- The application accepts articles of any length

### 3. **Analyze**

- Click the "🔍 Analyze Text" button
- The system will process the text through all ML models

### 4. **View Results**

- **Analysis Results Tab**: Shows the final verdict and confidence level
- **Visualizations Tab**: Interactive charts comparing model predictions
- **Model Details Tab**: Information about each ML model

### 5. **Review History**

- All analyses are saved in the history panel
- Double-click any history item to reload and review

## 📊 Understanding the Results

### Verdict Types

- ** APPEARS TO BE REAL NEWS**: High confidence in authenticity
- ** FAKE NEWS DETECTED**: High confidence in detecting misinformation

### Confidence Levels

- **High (80-100%)**: Very reliable prediction
- **Medium (60-79%)**: Moderately reliable prediction
- **Low (0-59%)**: Less reliable, manual verification recommended

### Model Predictions Table

Shows individual results from each model:

- **Model**: Name of the ML algorithm
- **Prediction**: REAL or FAKE classification
- **Confidence**: Certainty level of the prediction

## 🔧 Model Training (Advanced)

The application comes with pre-trained models, but you can retrain with your own data:

1. **Prepare Dataset**: CSV with 'text' and 'label' columns
2. **Feature Extraction**: TF-IDF vectorization with 5000 features
3. **Model Training**: Train Naive Bayes, SVM, and Neural Network
4. **Model Persistence**: Save using pickle for loading in GUI

##  Limitations & Considerations

- **Training Data**: Model accuracy depends on training data quality
- **Language**: Optimized for English text
- **Context**: Cannot verify factual claims, only detects linguistic patterns
- **Bias**: May reflect biases present in training data
- **Updates**: Models should be retrained periodically with fresh data

##  Creating Executable

Convert to standalone executable using PyInstaller:

```bash
# Install PyInstaller
pip install pyinstaller

# Create executable
pyinstaller --onefile --windowed --name "FakeNewsDetector" fake_news_detector_app.py

# With models folder
pyinstaller --onefile --windowed --name "FakeNewsDetector" --add-data "models;models" fake_news_detector_app.py
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Contact

**OLADELE HIKMAH** - [@Hikmah Oladele](https://github.com/Hikmaholadele)


##  Acknowledgments

- scikit-learn community for ML algorithms
- Tkinter for GUI framework
- matplotlib/seaborn for visualizations
- Open source community for inspiration and support

##  Future Enhancements

- [ ] Support for multiple languages
- [ ] Real-time news feed integration
- [ ] Web-based interface
- [ ] Advanced NLP techniques (BERT, transformers)
- [ ] Database integration for large-scale analysis
- [ ] API endpoint for external integration
- [ ] Mobile application version

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Always verify important news through multiple reliable sources. The predictions are based on linguistic patterns and may not always be accurate.
