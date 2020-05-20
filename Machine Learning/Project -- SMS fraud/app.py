from flask import Flask, render_template, redirect, url_for, request, make_response, jsonify
from sklearn.externals import joblib
import numpy as np
import requests
import json

import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


app = Flask(__name__)

# Chargement des fichiers avec les objets sklearn pour le preprocessing et le modèle
imputer = joblib.load("imputer.pkl")
featureencoder = joblib.load("featureencoder.pkl")
labelencoder = joblib.load("labelencoder.pkl")
classifier = joblib.load('classifier.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method=='POST':
        # Recover informations from html form
        data = dict(request.form.items())

        country = data["Country"]
        # handling missing fields for age and salary
        try:
            age = float(data["Age"])
        except ValueError:
            age = None
        try:
            salary = float(data["Salary"])
        except ValueError:
            salary = None

        # Create DataFrame with columns in the same order as in src/Data.csv
        #@TODO
        info = pd.DataFrame({'country': [country], 'age': [age], 'salary': [salary]})
        idx = 0
        numeric_features = []
        numeric_indices = []
        categorical_features = []
        categorical_indices = []

        for i, t in info.dtypes.iteritems():
            if ('float' in str(t)) or ('int' in str(t)):
                numeric_features.append(i)
                numeric_indices.append(idx)
            else:
                categorical_features.append(i)
                categorical_indices.append(idx)

            idx = idx + 1

        # Convert dataframe to numpy array before using scikit-learn
        #@TODO
        X = info.values
        # Preprocessings : impute and scale/encode features
        #@TODO

        X[:,numeric_indices] = imputer.transform(X[:,numeric_indices])
        X = featureencoder.transform(info)

        # Prediction
        #@TODO

        prediction = classifier.predict(X)

        # Use labelencoder to translate prediction into 'yes' or 'no'
        #@TODO
        labelencoder = joblib.load('labelencoder.pkl')
        prediction_translated = labelencoder.inverse_transform(prediction)


    return render_template("predicted.html", text=prediction_translated)


if __name__ == '__main__':
    app.run(debug=True)
