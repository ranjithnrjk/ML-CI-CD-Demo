import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load model
model = joblib.load('model/best_model.pkl')

# Load iris dataset for feature info
iris = load_iris()
feature_names = iris.feature_names

st.title("Iris Classifier")

# Input features
st.write("Enter the features of the Iris flower:")
features = [st.number_input(name, min_value=0.0) for name in feature_names]

if st.button('Predict'):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    st.write(f"Predicted class: {iris.target_names[prediction][0]}")
