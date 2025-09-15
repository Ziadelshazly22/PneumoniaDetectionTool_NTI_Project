@echo off
REM Quick Start Script for Pneumonia Detection Project
REM Creates venv, installs requirements, and runs Streamlit app

python -m venv .env
call .env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app\app.py
