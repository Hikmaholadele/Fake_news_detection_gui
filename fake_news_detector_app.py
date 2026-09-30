import os
import pickle
import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinter.font import Font
import sys
import numpy
import numpy as np
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# Handle numpy compatibility issues for pickle loading
import sys

# Comprehensive numpy compatibility fix for different versions
def setup_numpy_compatibility():
    try:
        import numpy._core
    except ImportError:
        try:
            import numpy.core
            sys.modules['numpy._core'] = numpy.core
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            sys.modules['numpy._core.umath'] = numpy.core.umath
        except ImportError:
            # Fallback for older numpy versions
            sys.modules['numpy._core'] = numpy
            sys.modules['numpy._core.multiarray'] = numpy
            sys.modules['numpy._core.umath'] = numpy

setup_numpy_compatibility()

# Define preprocessing function at module level so it can be pickled/unpickled
def preprocess_text(text):
    """Preprocess text for fake news detection - enhanced version matching Kaggle"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs, emails, HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|<.*?>', '', text)
    
    # Remove special characters and numbers, but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    
    # Remove very short words (less than 2 characters)
    words = text.split()
    words = [word for word in words if len(word) >= 2]
    text = ' '.join(words)
    
    return text

# Set matplotlib style for plots (equivalent to seaborn whitegrid)
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'white'
plt.rcParams['axes.facecolor'] = '#f0f0f0'

class FakeNewsDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Fake News Detection System")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate responsive window size (70% of screen size, more compact)
        window_width = min(1200, int(screen_width * 0.7))
        window_height = min(750, int(screen_height * 0.7))
        
        # Calculate center position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Set geometry with center positioning
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(900, 550)  # Reduced minimum size for smaller screens
        
        # Configure the style
        self.style = ttk.Style()
        self.configure_styles()
        
        self.primary_color = "#15803d"      # Green-700 - trustworthy primary
        self.secondary_color = "#84cc16"    # Bright green - accent
        self.bg_color = "#ffffff"           # Clean white background
        self.card_bg = "#f0fdf4"           # Light green for cards
        self.text_color = "#374151"         # Dark gray for text
        self.fake_color = "#be123c"         # Rose-700 for fake news
        self.real_color = "#15803d"         # Green-700 for real news
        self.border_color = "#d1d5db"       # Light gray borders
        self.muted_color = "#6b7280"        # Muted text
        
        # Set app icon if available
        try:
            self.root.iconbitmap("assets/app_icon.ico")
        except:
            pass  # Icon not available, continue without it
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
        
        # Initialize models and components
        self.models = {}
        self.vectorizer = None
        self.scaler = None
        self.preprocess_text = None
        self.best_model_name = None
        
        # Load models
        self.load_models()
        
        # Create the main layout
        self.create_layout()
        
    def configure_styles(self):
        """Configure ttk styles for a modern look"""
        try:
            self.root.tk.call('source', 'azure.tcl')
            self.style.theme_use('azure')
        except:
            available_themes = self.style.theme_names()
            if 'clam' in available_themes:
                self.style.theme_use('clam')
        
        # Configure modern styles
        self.style.configure('TFrame', background='#ffffff')
        self.style.configure('Card.TFrame', background='#f0fdf4', relief='solid', borderwidth=1)
        
        # Button styles
        self.style.configure('Primary.TButton', 
                           font=('Segoe UI', 11, 'bold'),
                           background='#15803d',
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none')
        self.style.map('Primary.TButton',
                      background=[('active', '#166534')])
        
        self.style.configure('Secondary.TButton', 
                           font=('Segoe UI', 11),
                           background='#84cc16',
                           foreground='#374151',
                           borderwidth=0,
                           focuscolor='none')
        self.style.map('Secondary.TButton',
                      background=[('active', '#65a30d')])
        
        # Label styles
        self.style.configure('TLabel', background='#ffffff', font=('Segoe UI', 10), foreground='#374151')
        self.style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground='#15803d')
        self.style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'), foreground='#374151')
        self.style.configure('Subheader.TLabel', font=('Segoe UI', 12, 'bold'), foreground='#374151')
        self.style.configure('Result.TLabel', font=('Segoe UI', 12), foreground='#374151')
        self.style.configure('Fake.TLabel', foreground='#be123c', font=('Segoe UI', 18, 'bold'))
        self.style.configure('Real.TLabel', foreground='#15803d', font=('Segoe UI', 18, 'bold'))
        self.style.configure('Muted.TLabel', foreground='#6b7280', font=('Segoe UI', 9))
        
        # Card label styles
        self.style.configure('Card.TLabel', background='#f0fdf4', font=('Segoe UI', 10), foreground='#374151')
        self.style.configure('CardHeader.TLabel', background='#f0fdf4', font=('Segoe UI', 11, 'bold'), foreground='#374151')
        
        # Notebook styles
        self.style.configure('TNotebook', background='#ffffff', borderwidth=0)
        self.style.configure('TNotebook.Tab', 
                           font=('Segoe UI', 12),
                           padding=[20, 10],
                           background='#f9fafb',
                           foreground='#374151')
        self.style.map('TNotebook.Tab',
                      background=[('selected', '#15803d')],
                      foreground=[('selected', 'white')])
        
        # LabelFrame styles
        self.style.configure('TLabelframe', background='#ffffff', borderwidth=1, relief='solid')
        self.style.configure('TLabelframe.Label', background='#ffffff', font=('Segoe UI', 12, 'bold'), foreground='#15803d')

    def load_models(self):
        """Load all model components from files"""
        # Handle both development and executable paths
        if getattr(sys, 'frozen', False):
            # Running as executable
            models_dir = os.path.join(sys._MEIPASS, "new_models")
        else:
            # Running as script
            models_dir = "new_models"
        
        try:
            # Add debug prints to diagnose issues
            print(f"Starting model loading from directory: {models_dir}")
            print(f"Directory exists: {os.path.exists(models_dir)}")
            
            if os.path.exists(models_dir):
                # Try to load the actual trained models
                print("Attempting to load actual trained models")
                
                # Load vectorizer (suppress sklearn version warnings)
                vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
                if os.path.exists(vectorizer_path):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with open(vectorizer_path, 'rb') as f:
                            self.vectorizer = pickle.load(f)
                    print("Vectorizer loaded successfully")
                
                # Load preprocessing function (use module-level function as fallback)
                # Note: Enhanced models don't have separate preprocessing function
                print("Using module-level preprocessing function")
                self.preprocess_text = preprocess_text
                
                # Load models with special handling for BPNN numpy issues
                model_files = {
                    'Naive Bayes': "naive_bayes.pkl",
                    'SVM': "svm.pkl", 
                    'BPNN': "bpnn.pkl"
                }
                
                self.models = {}
                for model_name, filename in model_files.items():
                    model_path = os.path.join(models_dir, filename)
                    if os.path.exists(model_path):
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                
                                # Special handling for BPNN model with numpy random state issues
                                if model_name == 'BPNN':
                                    # Try to handle numpy random state compatibility
                                    try:
                                        with open(model_path, 'rb') as f:
                                            model = pickle.load(f)
                                            # Reset random state to avoid numpy compatibility issues
                                            if hasattr(model, 'random_state'):
                                                model.random_state = 42
                                            self.models[model_name] = model
                                    except Exception as bpnn_error:
                                        print(f"BPNN pickle loading failed: {bpnn_error}")
                                        # Create a new BPNN model as fallback
                                        from sklearn.neural_network import MLPClassifier
                                        self.models[model_name] = MLPClassifier(
                                            hidden_layer_sizes=(100,), 
                                            max_iter=300, 
                                            random_state=42,
                                            solver='adam',
                                            alpha=0.0001
                                        )
                                        print(f"Created new {model_name} model as fallback")
                                        continue
                                else:
                                    with open(model_path, 'rb') as f:
                                        self.models[model_name] = pickle.load(f)
                                        
                            print(f"{model_name} model loaded successfully")
                        except Exception as e:
                            print(f"Failed to load {model_name} model: {e}")
                            # For critical models, create fallbacks
                            if model_name == 'Naive Bayes':
                                from sklearn.naive_bayes import MultinomialNB
                                self.models[model_name] = MultinomialNB()
                                print(f"Created new {model_name} model as fallback")
                            elif model_name == 'SVM':
                                from sklearn.svm import SVC
                                self.models[model_name] = SVC(probability=True, random_state=42)
                                print(f"Created new {model_name} model as fallback")
                
                # Load scaler
                scaler_path = os.path.join(models_dir, "scaler.pkl")
                if os.path.exists(scaler_path):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            with open(scaler_path, 'rb') as f:
                                self.scaler = pickle.load(f)
                        print("Scaler loaded successfully")
                    except Exception as e:
                        print(f"Failed to load scaler: {e}")
                        # Create a default scaler
                        from sklearn.preprocessing import StandardScaler
                        self.scaler = StandardScaler()
                else:
                    from sklearn.preprocessing import StandardScaler
                    self.scaler = StandardScaler()
                
                self.best_model_name = 'BPNN'
                
                if self.models and self.vectorizer:
                    print("All models loaded successfully from pickle files")
                    return
            
            # Fallback: create new models for demonstration
            print("Creating new models for demonstration")
            
            # Create TF-IDF vectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000)
            
            # Create a preprocessing function
            def preprocess_demo(text):
                if not text:
                    return ""
                text = text.lower()
                text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+|<.*?>', '', text)
                text = re.sub(r'[^a-zA-Z\s]', '', text)
                text = ' '.join(text.split())
                return text
            
            self.preprocess_text = preprocess_demo
            
            # Create basic models
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.svm import SVC
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
            
            self.models = {
                'Naive Bayes': MultinomialNB(),
                'SVM': SVC(probability=True),
                'BPNN': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
            }
            
            self.scaler = StandardScaler()
            
            # Set the best model
            self.best_model_name = 'BPNN'
            
            print("Models created successfully for demonstration")
            return
            
        except Exception as e:
            error_message = f"Failed to load models: {str(e)}"
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error Loading Models", 
                                 f"{error_message}\n\n"
                                 f"Make sure model files are in the /models directory.")
            # Create empty dictionaries to prevent errors
            self.models = {}
            self.vectorizer = None
            self.scaler = None
            self.preprocess_text = None

    def create_layout(self):
        """Create the main application layout"""
        
        # Main container with padding (directly in root)
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header section
        header_frame = tk.Frame(main_container, bg=self.bg_color)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # App title and subtitle
        title_label = ttk.Label(
            header_frame, 
            text="ML Fake News Detection", 
            style='Title.TLabel'
        )
        title_label.pack(anchor=tk.W)
        
        subtitle_label = ttk.Label(
            header_frame, 
            text="Advanced machine learning system for news authenticity analysis", 
            style='Muted.TLabel'
        )
        subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Status indicator
        if self.models:
            status_frame = tk.Frame(header_frame, bg=self.bg_color)
            status_frame.pack(anchor=tk.E, side=tk.RIGHT)
            
            status_dot = tk.Label(status_frame, text="●", fg=self.real_color, bg=self.bg_color, font=('Segoe UI', 16))
            status_dot.pack(side=tk.LEFT)
            
            status_text = ttk.Label(status_frame, text=f"Models Ready • Best: {self.best_model_name}", style='Muted.TLabel')
            status_text.pack(side=tk.LEFT, padx=(5, 0))
        
        # Main content area
        content_frame = tk.Frame(main_container, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input and controls (responsive width)
        screen_width = self.root.winfo_screenwidth()
        panel_width = min(480, int(screen_width * 0.32))  # Reduced from 550 to 480, 32% of screen
        left_panel = tk.Frame(content_frame, bg=self.bg_color, width=panel_width)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Input card
        input_card = ttk.Frame(left_panel, style='Card.TFrame', padding=15)
        input_card.pack(fill=tk.X, pady=(0, 12))  # Reduced padding and spacing
        
        input_header = ttk.Label(input_card, text="News Text Analysis", style='CardHeader.TLabel')
        input_header.pack(anchor=tk.W, pady=(0, 12))
        
        input_label = ttk.Label(input_card, text="Enter or paste news text to analyze:", style='Card.TLabel')
        input_label.pack(anchor=tk.W, pady=(0, 6))
        
        # Text input with modern styling
        text_frame = tk.Frame(input_card, bg='#f0fdf4')
        text_frame.pack(fill=tk.X, pady=(0, 12))
        
        self.text_input = scrolledtext.ScrolledText(
            text_frame, 
            height=5,  # Reduced from 6 to 5 to save space
            wrap=tk.WORD,
            font=('Segoe UI', 10),
            background="white",
            foreground=self.text_color,
            borderwidth=1,
            relief="solid",
            highlightthickness=1,
            highlightcolor=self.primary_color,
            selectbackground=self.secondary_color,
            selectforeground=self.text_color
        )
        self.text_input.pack(fill=tk.X, padx=1, pady=1)
        
        # Button row
        button_frame = tk.Frame(input_card, bg='#f0fdf4')
        button_frame.pack(fill=tk.X)
        
        self.analyze_btn = ttk.Button(
            button_frame, 
            text="🔍 Analyze Text", 
            command=self.analyze_text,
            style='Primary.TButton'
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(
            button_frame, 
            text="Clear", 
            command=lambda: self.text_input.delete(1.0, tk.END),
            style='Secondary.TButton'
        )
        self.clear_btn.pack(side=tk.LEFT)
        
        # Threshold controls card
        threshold_card = ttk.Frame(left_panel, style='Card.TFrame', padding=12)  # Reduced padding from 15 to 12
        threshold_card.pack(fill=tk.X, pady=(0, 12))  # Reduced from 15 to 12
        
        threshold_header = ttk.Label(threshold_card, text="Classification Thresholds", style='CardHeader.TLabel')
        threshold_header.pack(anchor=tk.W, pady=(0, 8))  # Reduced from 10 to 8
        
        # Initialize threshold values
        self.thresholds = {
            'Naive Bayes': tk.DoubleVar(value=0.5),
            'SVM': tk.DoubleVar(value=0.5),
            'BPNN': tk.DoubleVar(value=0.5)
        }
        
        # Create threshold sliders for each model
        for model_name in ['Naive Bayes', 'SVM', 'BPNN']:
            model_frame = tk.Frame(threshold_card, bg='#f0fdf4')
            model_frame.pack(fill=tk.X, pady=(0, 6))  # Reduced from 8 to 6
            
            # Model label
            model_label = ttk.Label(
                model_frame, 
                text=f"{model_name}:", 
                style='Card.TLabel'
            )
            model_label.pack(anchor=tk.W)
            
            # Threshold value display and slider container
            threshold_container = tk.Frame(model_frame, bg='#f0fdf4')
            threshold_container.pack(fill=tk.X, pady=(2, 0))  # Reduced from 3 to 2
            
            # Threshold value display
            threshold_value_label = ttk.Label(
                threshold_container,
                text=f"{self.thresholds[model_name].get():.2f}",
                style='Muted.TLabel'
            )
            threshold_value_label.pack(side=tk.RIGHT, padx=(10, 0))
            
            # Threshold slider
            threshold_slider = ttk.Scale(
                threshold_container,
                from_=0.1,
                to=0.9,
                variable=self.thresholds[model_name],
                orient=tk.HORIZONTAL,
                command=lambda val, label=threshold_value_label, var=self.thresholds[model_name]: self.update_threshold_display(val, label, var)
            )
            threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Auto-update checkbox
        self.auto_update_var = tk.BooleanVar(value=False)
        auto_update_cb = ttk.Checkbutton(
            threshold_card,
            text="Auto-update predictions",
            variable=self.auto_update_var,
            style='Card.TCheckbutton'
        )
        auto_update_cb.pack(anchor=tk.W, pady=(6, 0))  # Reduced from 8 to 6
        
        # Right panel - Results and visualizations
        right_panel = tk.Frame(content_frame, bg=self.bg_color)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results notebook with modern tabs
        self.result_notebook = ttk.Notebook(right_panel)
        self.result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Results tab
        self.results_tab = tk.Frame(self.result_notebook, bg=self.bg_color)
        self.result_notebook.add(self.results_tab, text="📊 Analysis Results")
        
        # Create results layout
        self.create_results_layout()
        
        # Visualizations tab
        self.viz_tab = tk.Frame(self.result_notebook, bg=self.bg_color)
        self.result_notebook.add(self.viz_tab, text="📈 Visualizations")
        
        # Create visualization layout
        self.create_visualization_layout()
        
        # Models tab
        self.models_tab = tk.Frame(self.result_notebook, bg=self.bg_color)
        self.result_notebook.add(self.models_tab, text="🤖 Model Details")
        
        # Create models layout
        self.create_models_layout()
        
        # Status bar
        status_bar = tk.Frame(main_container, bg=self.border_color, height=1)
        status_bar.pack(fill=tk.X, pady=(15, 0))
        
        status_container = tk.Frame(main_container, bg=self.bg_color)
        status_container.pack(fill=tk.X, pady=(8, 0))
        
        self.status_label = ttk.Label(
            status_container, 
            text="Ready to analyze news text. Enter text above and click 'Analyze Text'.",
            style='Muted.TLabel'
        )
        self.status_label.pack(side=tk.LEFT)
        
        version_label = ttk.Label(
            status_container, 
            text="v2.0.0 • Enhanced UI",
            style='Muted.TLabel'
        )
        version_label.pack(side=tk.RIGHT)

    def create_results_layout(self):
        """Create the results tab layout"""
        results_container = tk.Frame(self.results_tab, bg=self.bg_color)
        results_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Main verdict card
        verdict_card = ttk.Frame(results_container, style='Card.TFrame', padding=20)
        verdict_card.pack(fill=tk.X, pady=(0, 15))
        
        # Verdict header
        verdict_header_frame = tk.Frame(verdict_card, bg='#f0fdf4')
        verdict_header_frame.pack(fill=tk.X, pady=(0, 15))
        
        verdict_title = ttk.Label(verdict_header_frame, text="Analysis Verdict", style='CardHeader.TLabel')
        verdict_title.pack(side=tk.LEFT)
        
        self.result_time = ttk.Label(verdict_header_frame, text="", style='Muted.TLabel')
        self.result_time.pack(side=tk.RIGHT)
        
        # Main verdict display
        self.verdict_result = ttk.Label(
            verdict_card, 
            text="No analysis performed yet", 
            style='Real.TLabel'
        )
        self.verdict_result.pack(pady=(0, 15))
        
        # Confidence display
        confidence_frame = tk.Frame(verdict_card, bg='#f0fdf4')
        confidence_frame.pack(fill=tk.X)
        
        confidence_label = ttk.Label(confidence_frame, text="Confidence Level", style='Card.TLabel')
        confidence_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Progress bar container
        progress_container = tk.Frame(confidence_frame, bg='#f0fdf4')
        progress_container.pack(fill=tk.X, pady=(0, 5))
        
        self.confidence_bar = ttk.Progressbar(
            progress_container, 
            orient=tk.HORIZONTAL, 
            length=300, 
            mode='determinate',
            style='TProgressbar'
        )
        self.confidence_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.confidence_text = ttk.Label(
            progress_container, 
            text="0%", 
            style='CardHeader.TLabel'
        )
        self.confidence_text.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Model results section
        models_results_card = ttk.Frame(results_container, style='Card.TFrame', padding=15)
        models_results_card.pack(fill=tk.BOTH, expand=True)
        
        models_header = ttk.Label(models_results_card, text="Individual Model Predictions", style='CardHeader.TLabel')
        models_header.pack(anchor=tk.W, pady=(0, 12))
        
        # Create model results table
        self.create_model_results_table(models_results_card)

    def create_model_results_table(self, parent):
        """Create a modern table for model results"""
        table_frame = tk.Frame(parent, bg='#f0fdf4')
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Table headers
        headers = ["Model", "Prediction", "Confidence", "Threshold"]
        header_frame = tk.Frame(table_frame, bg='white', relief='solid', borderwidth=1)
        header_frame.pack(fill=tk.X, pady=(0, 1))
        
        for i, header in enumerate(headers):
            header_label = tk.Label(
                header_frame, 
                text=header, 
                font=('Segoe UI', 10, 'bold'),
                bg='white',
                fg=self.primary_color,
                padx=8,
                pady=8
            )
            header_label.grid(row=0, column=i, sticky='ew', padx=0, pady=0)
            header_frame.grid_columnconfigure(i, weight=1, uniform="col")
        
        # Model rows
        self.model_result_rows = {}
        models = ["Naive Bayes", "SVM", "BPNN", "Weighted Ensemble"]
        
        for idx, model_name in enumerate(models):
            row_frame = tk.Frame(table_frame, bg='white', relief='solid', borderwidth=1)
            row_frame.pack(fill=tk.X, pady=(0, 1))
            
            # Model name
            model_label = tk.Label(
                row_frame, text=model_name, 
                font=('Segoe UI', 9, 'bold'),
                bg='white', fg=self.text_color,
                padx=8, pady=6
            )
            model_label.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
            
            # Prediction
            prediction_label = tk.Label(
                row_frame, text="-", 
                font=('Segoe UI', 9, 'bold'),
                bg='white', fg=self.muted_color,
                padx=8, pady=6
            )
            prediction_label.grid(row=0, column=1, sticky='ew', padx=0, pady=0)
            
            # Confidence
            confidence_label = tk.Label(
                row_frame, text="-", 
                font=('Segoe UI', 9),
                bg='white', fg=self.text_color,
                padx=8, pady=6
            )
            confidence_label.grid(row=0, column=2, sticky='ew', padx=0, pady=0)
            
            # Threshold
            threshold_label = tk.Label(
                row_frame, text="-", 
                font=('Segoe UI', 9),
                bg='white', fg=self.muted_color,
                padx=8, pady=6
            )
            threshold_label.grid(row=0, column=3, sticky='ew', padx=0, pady=0)
            
            # Configure column weights with uniform sizing
            for i in range(4): # Updated to 4 columns
                row_frame.grid_columnconfigure(i, weight=1, uniform="col")
            
            # Store references
            self.model_result_rows[model_name] = {
                'prediction': prediction_label,
                'confidence': confidence_label,
                'threshold': threshold_label
            }

    def create_visualization_layout(self):
        """Create the visualization tab layout"""
        viz_container = tk.Frame(self.viz_tab, bg=self.bg_color)
        viz_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        viz_header = ttk.Label(viz_container, text="Model Performance Visualization", style='Header.TLabel')
        viz_header.pack(anchor=tk.W, pady=(0, 15))
        
        # Matplotlib figure
        self.fig = plt.Figure(figsize=(9, 6), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_models_layout(self):
        """Create the models information tab"""
        models_container = tk.Frame(self.models_tab, bg=self.bg_color)
        models_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        models_header = ttk.Label(models_container, text="Machine Learning Models", style='Header.TLabel')
        models_header.pack(anchor=tk.W, pady=(0, 15))
        
        # Model cards
        models_info = [
            ("Naive Bayes", "Probabilistic classifier based on Bayes' theorem with strong independence assumptions", "Fast, simple, works well with small datasets"),
            ("Support Vector Machine", "Finds optimal hyperplane to separate fake and real news in high-dimensional space", "Effective for text classification, handles high dimensions well"),
            ("Neural Network (BPNN)", "Multi-layer perceptron with backpropagation learning algorithm", "Can learn complex patterns, often provides best accuracy"),
            ("Weighted Ensemble", "Combines predictions from all models using weighted voting", "Leverages strengths of all models for robust predictions")
        ]
        
        for name, description, advantages in models_info:
            model_card = ttk.Frame(models_container, style='Card.TFrame', padding=15)
            model_card.pack(fill=tk.X, pady=(0, 12))
            
            model_name = ttk.Label(model_card, text=name, style='CardHeader.TLabel')
            model_name.pack(anchor=tk.W, pady=(0, 8))
            
            model_desc = ttk.Label(model_card, text=description, style='Card.TLabel', wraplength=600)
            model_desc.pack(anchor=tk.W, pady=(0, 5))
            
            model_adv = ttk.Label(model_card, text=f"Advantages: {advantages}", style='Muted.TLabel', wraplength=600)
            model_adv.pack(anchor=tk.W)

    def load_text(self, text):
        """Load text into the input field"""
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(tk.END, text)
        self.status_label.config(text=f"Loaded example text ({len(text)} characters)")

    def update_threshold_display(self, value, label, var):
        """Update threshold display and optionally re-analyze if auto-update is enabled"""
        label.config(text=f"{float(value):.2f}")
        
        # If auto-update is enabled and there's text to analyze
        if self.auto_update_var.get() and self.text_input.get(1.0, tk.END).strip():
            # Delay the analysis slightly to avoid rapid-fire updates
            self.root.after(300, self.analyze_text)

    def analyze_text(self):
        """Analyze the entered text for fake news"""
        # Get text from input
        text = self.text_input.get(1.0, tk.END).strip()
        
        if not text:
            messagebox.showinfo("Input Required", "Please enter some text to analyze.")
            return
        
        # Check if models are loaded
        print(f"Models loaded: {len(self.models)}")
        print(f"Models available: {list(self.models.keys()) if self.models else 'None'}")
        print(f"Vectorizer loaded: {self.vectorizer is not None}")
        
        if not self.models or not self.vectorizer:
            messagebox.showerror("Models Not Loaded", 
                               "Models could not be loaded. Please check the models directory.\n\n"
                               "Note: Try running the application from the command line to see detailed error messages.")
            return
        
        # Update status
        self.status_label.config(text="Analyzing text... Please wait.")
        
        # Use threading to prevent UI freeze during analysis
        threading.Thread(target=self._analyze_text_thread, args=(text,), daemon=True).start()

    def _analyze_text_thread(self, text):
        """Run analysis in a separate thread to prevent UI freeze"""
        try:
            # Ensure we have models loaded
            if not self.models or not self.vectorizer:
                raise ValueError("Models not properly loaded. Please restart the application.")
                
            # Preprocess the text
            try:
                processed_text = self.clean_text(text)
                print(f"Text preprocessed successfully: '{processed_text[:50]}...'")
            except Exception as preprocess_error:
                print(f"Error in preprocessing: {str(preprocess_error)}")
                # Fall back to basic preprocessing
                processed_text = self.basic_preprocess(text)
                print(f"Used fallback preprocessing: '{processed_text[:50]}...'")
            
            # Transform text using the trained vectorizer
            try:
                text_tfidf = self.vectorizer.transform([processed_text])
                print(f"Text vectorized successfully. Shape: {text_tfidf.shape}")
            except Exception as e:
                print(f"Error vectorizing text: {e}")
                raise ValueError(f"Failed to vectorize text: {e}")
            
            # Store all results
            all_predictions = {}
            
            # Get actual predictions from trained models
            for model_name, model in self.models.items():
                try:
                    # Get current threshold for this model
                    threshold = self.thresholds.get(model_name, tk.DoubleVar(value=0.5)).get()
                    
                    # Get probability predictions from the model
                    if hasattr(model, 'predict_proba'):
                        # For models that support probability prediction
                        probabilities = model.predict_proba(text_tfidf)[0]
                        fake_prob = probabilities[0]  # Class 0 = Fake
                        real_prob = probabilities[1]  # Class 1 = Real
                    else:
                        # For models that don't support predict_proba, use decision_function or predict
                        prediction = model.predict(text_tfidf)[0]
                        if hasattr(model, 'decision_function'):
                            # Use decision function to estimate confidence
                            decision_score = model.decision_function(text_tfidf)[0]
                            # Convert decision score to probability-like values
                            confidence_score = 1 / (1 + np.exp(-decision_score))  # Sigmoid
                            if prediction == 1:  # Real
                                real_prob = confidence_score
                                fake_prob = 1 - confidence_score
                            else:  # Fake
                                fake_prob = confidence_score
                                real_prob = 1 - confidence_score
                        else:
                            # Fallback: assign moderate confidence
                            if prediction == 1:
                                real_prob = 0.75
                                fake_prob = 0.25
                            else:
                                fake_prob = 0.75
                                real_prob = 0.25
                    
                    # Apply threshold for prediction (threshold is for real news)
                    prediction = 1 if real_prob >= threshold else 0
                    
                    # Calculate result and confidence
                    result = "REAL" if prediction == 1 else "FAKE"
                    confidence = real_prob if prediction == 1 else fake_prob
                    
                    all_predictions[model_name] = {
                        'prediction': result,
                        'confidence': confidence,
                        'fake_probability': fake_prob,
                        'real_probability': real_prob,
                        'threshold': threshold,
                        'threshold_based': True
                    }
                    
                    print(f"{model_name} prediction: {result} (confidence: {confidence:.2f}, threshold: {threshold:.2f})")
                    
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    # Fallback for this specific model
                    all_predictions[model_name] = {
                        'prediction': "UNKNOWN",
                        'confidence': 0.5,
                        'fake_probability': 0.5,
                        'real_probability': 0.5,
                        'threshold': threshold,
                        'threshold_based': True
                    }
            
            
            # Calculate weighted prediction
            weighted_fake_prob = 0
            weighted_real_prob = 0
            total_weight = len(self.models)  # Equal weights for simplicity
            
            for model_name, pred_data in all_predictions.items():
                weighted_fake_prob += pred_data['fake_probability']
                weighted_real_prob += pred_data['real_probability']
            
            weighted_fake_prob /= total_weight
            weighted_real_prob /= total_weight
            weighted_prediction = "REAL" if weighted_real_prob > weighted_fake_prob else "FAKE"
            weighted_confidence = max(weighted_fake_prob, weighted_real_prob)
            
            # Add ensemble result
            all_predictions["Weighted Ensemble"] = {
                'prediction': weighted_prediction,
                'confidence': weighted_confidence,
                'fake_probability': weighted_fake_prob,
                'real_probability': weighted_real_prob
            }
            
            # Record timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Store the analysis result
            analysis_result = {
                "text": text,
                "timestamp": timestamp,
                "verdict": weighted_prediction,
                "confidence": weighted_confidence,
                "model_predictions": all_predictions
            }
            
            # Store last analysis
            self.last_analysis = analysis_result
            
            # Update UI with results
            self.root.after(0, lambda: self._update_ui_with_results(analysis_result))
            
        except Exception as e:
            # Handle any errors
            error_message = str(e)
            self.root.after(0, lambda: self._show_error(error_message))

    def _update_ui_with_results(self, analysis_result):
        """Update the UI with analysis results"""
        verdict = analysis_result["verdict"]
        confidence = analysis_result["confidence"]
        
        # Update timestamp
        self.result_time.config(text=f"Analyzed: {analysis_result['timestamp']}")
        
        # Update verdict with enhanced styling
        if verdict == "FAKE":
            self.verdict_result.config(
                text="⚠️ FAKE NEWS DETECTED", 
                style='Fake.TLabel'
            )
        else:
            self.verdict_result.config(
                text="✅ APPEARS TO BE REAL NEWS", 
                style='Real.TLabel'
            )
            
        # Animate confidence bar
        current_value = self.confidence_bar['value']
        target_value = confidence * 100
        
        def animate_progress(current, target, step=3):
            if abs(current - target) <= step:
                self.confidence_bar['value'] = target
                self.confidence_text.config(text=f"{confidence:.1%}")
                return
            
            if current < target:
                current += step
            else:
                current -= step
                
            self.confidence_bar['value'] = current
            self.confidence_text.config(text=f"{current/100:.1%}")
            self.root.after(15, lambda: animate_progress(current, target))
        
        animate_progress(current_value, target_value)
        
        # Update model results table
        model_predictions = analysis_result["model_predictions"]
        
        for model_name, row_data in self.model_result_rows.items():
            if model_name in model_predictions:
                pred_data = model_predictions[model_name]
                
                # Update prediction with color
                prediction = pred_data['prediction']
                if prediction == "FAKE":
                    row_data['prediction'].config(text="⚠️ FAKE", fg=self.fake_color)
                else:
                    row_data['prediction'].config(text="✅ REAL", fg=self.real_color)
                
                # Update other fields
                row_data['confidence'].config(text=f"{pred_data['confidence']:.1%}")
                
                # Update threshold (only for individual models, not ensemble)
                if model_name != "Weighted Ensemble":
                    threshold_val = pred_data.get('threshold', 0.5)
                    row_data['threshold'].config(text=f"{threshold_val:.2f}")
                else:
                    row_data['threshold'].config(text="N/A")
        
        # Create visualizations
        self.create_visualizations(model_predictions)
        
        # Switch to results tab
        self.result_notebook.select(0)
        
        # Update status
        confidence_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        self.status_label.config(text=f"Analysis complete: {verdict} news detected with {confidence_level.lower()} confidence ({confidence:.1%})")

    def create_visualizations(self, model_predictions):
        """Create and update visualizations based on predictions"""
        self.fig.clear()
        
        # Set figure style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1])
        ax1 = self.fig.add_subplot(gs[0, :])  # Top spans both columns
        ax2 = self.fig.add_subplot(gs[1, 0])  # Bottom left
        ax3 = self.fig.add_subplot(gs[1, 1])  # Bottom right
        
        # 1. Model comparison bar chart
        model_names = []
        fake_probs = []
        real_probs = []
        
        for model_name, pred_data in model_predictions.items():
            if model_name != "Weighted Ensemble":
                model_names.append(model_name.replace(" ", "\n"))
                fake_probs.append(pred_data['fake_probability'])
                real_probs.append(pred_data['real_probability'])
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, fake_probs, width, label='Fake Probability', 
                       color='#be123c', alpha=0.8, edgecolor='white', linewidth=1)
        bars2 = ax1.bar(x + width/2, real_probs, width, label='Real Probability', 
                       color='#15803d', alpha=0.8, edgecolor='white', linewidth=1)
        
        ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax1.set_title('Model Prediction Comparison', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, fontsize=10)
        ax1.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Confidence levels pie chart
        model_names_all = list(model_predictions.keys())
        confidences = [data['confidence'] for data in model_predictions.values()]
        
        colors = ['#be123c' if model_predictions[name]['prediction'] == "FAKE" else '#15803d' 
                 for name in model_names_all]
        
        wedges, texts, autotexts = ax2.pie(confidences, labels=model_names_all, autopct='%1.1f%%',
                                          colors=colors, startangle=90, textprops={'fontsize': 9})
        ax2.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        
        # 3. Ensemble result gauge
        ensemble_data = model_predictions["Weighted Ensemble"]
        fake_prob = ensemble_data['fake_probability']
        real_prob = ensemble_data['real_probability']
        
        # Create a simple gauge-like visualization
        categories = ['Fake\nProbability', 'Real\nProbability']
        values = [fake_prob, real_prob]
        colors_gauge = ['#be123c', '#15803d']
        
        bars = ax3.bar(categories, values, color=colors_gauge, alpha=0.8, 
                      edgecolor='white', linewidth=2)
        ax3.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax3.set_title('Ensemble Prediction', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2., value + 0.02,
                    f'{value:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def _show_error(self, error_message):
        """Show error message in the UI"""
        messagebox.showerror("Analysis Error", f"Error analyzing text: {error_message}")
        self.status_label.config(text=f"Error: {error_message}")

    def clean_text(self, text):
        """Try to use the loaded preprocessing function or fall back to basic cleaning"""
        try:
            if self.preprocess_text:
                return self.preprocess_text(text)
        except Exception as e:
            print(f"Error using loaded preprocessing function: {str(e)}")
            return self.basic_preprocess(text)
            
    def basic_preprocess(self, text):
        """Basic text preprocessing as a fallback"""
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
        
        print("Used basic preprocessing")
        return text

def main():
    root = tk.Tk()
    app = FakeNewsDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
