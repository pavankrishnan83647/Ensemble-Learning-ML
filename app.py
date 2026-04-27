import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")

st.title("Iris Prediction")

f1 = st.slider("Feature 1", 0.0, 10.0)
f2 = st.slider("Feature 2", 0.0, 10.0)
f3 = st.slider("Feature 3", 0.0, 10.0)
f4 = st.slider("Feature 4", 0.0, 10.0)

if st.button("Predict"):
    data = np.array([[f1, f2, f3, f4]])
    result = model.predict(data)
    st.success(f"Class: {result[0]}")