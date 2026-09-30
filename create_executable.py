# Package the application into an executable using pyinstaller

import os
import sys
from pathlib import Path
import shutil

def create_executable():
    """
    Create an executable file from the fake_news_detector_app.py
    """
    # Check if PyInstaller is installed
    try:
        import PyInstaller
        print("PyInstaller is installed!")
    except ImportError:
        print("PyInstaller is not installed. Installing now...")
        os.system("pip install pyinstaller")
        print("PyInstaller installed successfully!")
    
    # Create a directory for the spec file and build files
    build_dir = "build"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    # Create a spec file
    spec_content = """
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['fake_news_detector_app.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models')],  # Include models directory
    hiddenimports=['matplotlib.backends.backend_tkagg', 'sklearn.neighbors.typedefs', 'sklearn.neighbors.quad_tree', 'sklearn.tree._utils'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FakeNewsDetector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico' if os.path.exists('assets/app_icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FakeNewsDetector',
)
    """
    
    spec_file = os.path.join(build_dir, "fake_news_detector.spec")
    with open(spec_file, "w") as f:
        f.write(spec_content)
    
    # Create assets directory if it doesn't exist
    assets_dir = "assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    
    # Create a README file
    readme_content = """# Fake News Detector Application

This application uses multiple machine learning models (Naive Bayes, SVM, and Neural Network) to detect fake news.

## Features
- Multi-model analysis with confidence scores
- Visualization of prediction confidence
- History tracking of previous analyses
- Sample news texts for testing
- Detailed breakdown of individual model predictions

## How to Use
1. Launch the application by running the executable
2. Enter or paste news text in the input field
3. Click "Detect Fake News" to analyze
4. View the results in the Results tab
5. Check your analysis history in the History tab

## Requirements
All required libraries and dependencies are packaged with the executable.

## Models
The trained ML models are included in the `models` directory.
"""
    
    readme_file = os.path.join(build_dir, "README.md")
    with open(readme_file, "w") as f:
        f.write(readme_content)
    
    # Run PyInstaller
    print("\nBuilding executable with PyInstaller...")
    os.system(f"pyinstaller {spec_file}")
    
    print("\nExecutable creation completed!")
    print(f"The executable is located in the 'dist/FakeNewsDetector' directory")
    print("You can distribute the entire 'FakeNewsDetector' folder to users.")

if __name__ == "__main__":
    create_executable()
