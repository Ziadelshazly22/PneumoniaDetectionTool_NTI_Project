# Pneumonia Detection from Chest X-rays

This project provides an end-to-end AI solution for pneumonia detection using chest X-ray images. It includes:


## Directory Structure
```
notebooks/   # Jupyter/Colab notebooks for model training
app/         # Streamlit application code
models/      # Trained model weights
data/        # Sample data or instructions for dataset download
```

## Getting Started

### Quick Setup (Recommended)
1. Open PowerShell in the project directory.
2. Run:
	```powershell
	.\quickstart.bat
	```
	This will create a virtual environment, install dependencies, and launch the Streamlit app.

### Manual Setup
1. Create a Python virtual environment:
	```powershell
	python -m venv .env
	.\.env\Scripts\activate
	```
2. Install dependencies:
	```powershell
	pip install --upgrade pip
	pip install -r requirements.txt
	```
3. Run the Streamlit app:
	```powershell
	streamlit run app/app.py
	```

## Features

- File uploader for chest X-ray images
- Model inference and visualization
- Bilingual UI (English/Arabic)
- Dark/light mode toggle
- GradCAM for explainable AI

## License
MIT
