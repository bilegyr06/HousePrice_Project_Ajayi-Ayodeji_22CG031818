import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import joblib
import logging

# Initialize Flask App
app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = app.logger

# Load the trained pipeline
try:
    logger.info("Loading model pipeline...")
    model = joblib.load('./model/house_price_model.pkl')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles form submission, processes data, and returns prediction.
    """
    if not model:
        return render_template('index.html', error="Model not loaded. Please contact administrator.")

    try:
        # 1. Extract data from form
        # We must cast inputs to float, as HTML forms send strings
        form_data = request.form
        
        # 2. Create a DataFrame matching the training data structure exactly
        # The pipeline requires these specific column names
        input_data = pd.DataFrame({
            'longitude': [float(form_data['longitude'])],
            'latitude': [float(form_data['latitude'])],
            'housing_median_age': [float(form_data['housing_median_age'])],
            'total_rooms': [float(form_data['total_rooms'])],
            'total_bedrooms': [float(form_data['total_bedrooms'])],
            'population': [float(form_data['population'])],
            'households': [float(form_data['households'])],
            'median_income': [float(form_data['median_income'])],
            'ocean_proximity': [form_data['ocean_proximity']] # Categorical (string)
        })
        
        # 3. Make Prediction
        # The pipeline handles scaling and one-hot encoding automatically
        prediction = model.predict(input_data)[0]
        
        # 4. Return result
        return render_template(
            'index.html', 
            prediction_text=f"${prediction:,.2f}",
            input_data=form_data # Pass back input to keep form filled (optional)
        )

    except ValueError as ve:
        logger.error(f"Value Error: {ve}")
        return render_template('index.html', error="Invalid input. Please ensure all numeric fields contain numbers.")
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# API Endpoint (Optional, for programmatic access)
@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        json_data = request.get_json()
        input_df = pd.DataFrame([json_data])
        prediction = model.predict(input_df)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)