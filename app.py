# Importing related libraries and modules
import os
import re
import pickle

from flask import Flask, jsonify, render_template, request
from sqlalchemy import create_engine, and_
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from decimal import Decimal
from urllib.parse import unquote_plus
from urllib.parse import unquote
import sqlalchemy
from joblib import load
import numpy as np
#################################################
#Loading Machine learning model
#################################################
# Assuming your model is named 'model.sav'
# model_path = r'C:\Users\Simon\Desktop\Data Bootcamp\Personal Project\Cancer-classification\Machine Learning Models\Neural Networks\smote_nn_model'
# model_path = r'C:\Users\Simon\Desktop\Data Bootcamp\Personal Project\Cancer-classification\Machine Learning Models\Random Forrest\rf_model'
model_path = r'C:\Users\Simon\Desktop\Data Bootcamp\Personal Project\Cancer-classification\Machine Learning Models\XGBoost\XGBoost_model'
model = load(model_path)

#################################################
# Flask API App Setup
#################################################
engine = create_engine("postgresql://Sim0304:9MbTw8pcFKvy@ep-summer-mud-a7ecgtua.ap-southeast-2.aws.neon.tech/Cancer-class?sslmode=require")
Base = automap_base()
Base.prepare(autoload_with=engine)
# Create a Flask web application instance
app = Flask(__name__)

#################################################
# Flask API Routes
#################################################

############# Add Homepage Route ###############

@app.route("/")
def homepage():
    # Serve the homepage HTML webpage (homepage.html)
    return render_template("homepage.html")

############# Add visualization Route ###############
@app.route("/visualization")
def visualization():
    return render_template("visualization.html")

############# Add Route for Handling Form Submission - Machine Learning Model ###############
@app.route("/predict", methods=["GET"])
def predict():
    # Extracting data from the form submission
    radius1 = request.args.get('radius1', type=float)
    texture1 = request.args.get('texture1', type=float)
    perimeter1 = request.args.get('perimeter1', type=float)
    area1 = request.args.get('area1', type=float)
    smoothness1 = request.args.get('smoothness1', type=float)
    compactness1 = request.args.get('compactness1', type=float)
    concavity1 = request.args.get('concavity1', type=float)
    concave_points1 = request.args.get('concave_points1', type=float)
    symmetry1 = request.args.get('symmetry1', type=float)
    fractal_dimension1 = request.args.get('fractal_dimension1', type=float)
    radius2 = request.args.get('radius2', type=float)
    texture2 = request.args.get('texture2', type=float)
    perimeter2 = request.args.get('perimeter2', type=float)
    area2 = request.args.get('area2', type=float)
    smoothness2 = request.args.get('smoothness2', type=float)
    compactness2 = request.args.get('compactness2', type=float)
    concavity2 = request.args.get('concavity2', type=float)
    concave_points2 = request.args.get('concave_points2', type=float)
    symmetry2 = request.args.get('symmetry2', type=float)
    fractal_dimension2 = request.args.get('fractal_dimension2', type=float)
    radius3 = request.args.get('radius3', type=float)
    texture3 = request.args.get('texture3', type=float)
    perimeter3 = request.args.get('perimeter3', type=float)
    area3 = request.args.get('area3', type=float)
    smoothness3 = request.args.get('smoothness3', type=float)
    compactness3 = request.args.get('compactness3', type=float)
    concavity3 = request.args.get('concavity3', type=float)
    concave_points3 = request.args.get('concave_points3', type=float)
    symmetry3 = request.args.get('symmetry3', type=float)
    fractal_dimension3 = request.args.get('fractal_dimension3', type=float)

    # Prepare the feature vector according to the specified format
    features = np.array([radius1, texture1, perimeter1, area1, smoothness1, compactness1, concavity1, concave_points1, symmetry1, fractal_dimension1, 
                         radius2, texture2, perimeter2, area2, smoothness2, compactness2, concavity2, concave_points2, symmetry2, fractal_dimension2,
                         radius3, texture3, perimeter3, area3, smoothness3, compactness3, concavity3, concave_points3, symmetry3, fractal_dimension3]).reshape(1,-1)
    print(features)
    
    # Predicting the churn status
    prediction = model.predict(features)
    print(prediction)
    
    result = 'Benign' if prediction[0] == 0 else 'Malignant'
    print(result)

    # Determine result status for styling
    result_status = 'Benign' if prediction[0] == 0 else 'Malignant'

    # Return result to the same page or to a new prediction result page
        # Include the summary in the context
    # Prepare input summary
    input_summary = f"Radius Mean: {radius1}, Texture Mean: {texture1}, Perimeter Mean: {perimeter1}, Area Mean: {area1}, Smoothness Mean: {smoothness1}, Compactness Mean: {compactness1}, Concavity Mean: {concavity1}, Concave Points Mean: {concave_points1}, Symmetry Mean: {symmetry1}, Fractal Dimension Mean: {fractal_dimension1}"
        # Pass result_status to the template
    return render_template('homepage.html', prediction_text=f'Breast Cancer Classification: {result}',result_status=result_status)

    

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



 