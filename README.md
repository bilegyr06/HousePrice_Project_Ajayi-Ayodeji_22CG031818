# House Price Prediction Web App

This project is a Flask-based web application that predicts house prices using a trained machine learning model.

## Features
- Accepts house-related input values through a web form
- Uses a pre-trained model to generate a price prediction
- Renders results in a simple HTML interface

## Project Structure
- `app.py` – Flask application entry point
- `templates/` – HTML templates for the web UI
- `model/` – trained model and training script
- `requirements.txt` – Python dependencies

## Setup
1. Create and activate a virtual environment
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Install dependencies
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the app
   ```powershell
   python app.py
   ```
4. Open the app in your browser at:
   ```text
   http://127.0.0.1:5000/
   ```

## Notes
- The app expects the trained model file at `model/house_price_model.pkl`.
- The model was created using the `model/model_building.py` script.

## Deployment
The project is also compatible with Gunicorn for deployment:
```powershell
gunicorn app:app
```
