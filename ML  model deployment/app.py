import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load the trained model
model_rf = pickle.load(open("model_rf.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
stdScaler = pickle.load(open("stdScalerModel.pkl", "rb"))
columns_to_be_scaled = pickle.load(open('columns_to_be_scaled.pkl', "rb"))
oneHotEncoder = pickle.load(open("enc.pkl", "rb"))

# Create flask app
flask_app = Flask(__name__)

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Get input features from form
    distance = request.form['distance']
    source = request.form['source']
    destination = request.form['destination']
    year = 2023
    quarter = 2

    passengers = request.form['passengers']
    carrier_lg = request.form['carrier_lg']
    ms_lg = request.form['ms_lg']
    carrier_low = request.form['carrier_low']
    ms_low = request.form['ms_low']

    lat1 = request.form['lat1']
    long1 = request.form['long1']
    lat2 = request.form['lat2']
    long2 = request.form['long2']
    ############################
    # distance = 1203
    # source = 32467
    # destination = 34576
    # year = 2023
    # quarter = 2
    #
    # passengers = 203
    # carrier_lg = 'FL'
    # ms_lg = 0.29
    # carrier_low = 'FL'
    # ms_low = 0.29
    #
    # lat1 = 45
    # long1 = -93
    # lat2 = 43
    # long2 = -77
    ############################

    features = [year, quarter, source, destination, distance, passengers, carrier_lg, ms_lg,
                carrier_low, ms_low, lat1, long1, lat2, long2]

    cols = ['Year', 'quarter', 'citymarketid_1', 'citymarketid_2',
       'nsmiles', 'passengers', 'carrier_lg', 'large_ms',
       'carrier_low', 'lf_ms', 'Latitude1', 'Longitude1', 'Latitude2', 'Longitude2']

    # Create a dataframe with the input features and get the list of model features
    df = pd.DataFrame([features], columns=cols)
    df[columns_to_be_scaled] = stdScaler.transform(df[columns_to_be_scaled])
    encoded_cols = oneHotEncoder.transform(df[['carrier_lg', 'carrier_low']])
    df_encoded = pd.concat([df, pd.DataFrame(encoded_cols.toarray())], axis='columns')
    df = df_encoded.drop(['carrier_lg', 'carrier_low'], axis='columns')
    X_test_pca = pca.transform(df.to_numpy())
    prediction = model_rf.predict(X_test_pca)

    return render_template("index.html", prediction_text="Predicted flight fare is $. {:.2f}".format(prediction[0]))

if __name__ == "__main__":
    flask_app.run(debug=True)