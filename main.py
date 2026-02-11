import streamlit as st
import pickle
import numpy as np


@st.cache_resource
def load_model():
    with open("model/random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

model, scaler = load_model()


st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")

feature1 = st.selectbox("Pclass (Ticket class)", options=[1, 2, 3])
gender = st.selectbox("Sex", options=["male", "female"])
feature2 = 0 if gender == "male" else 1
feature3 = st.number_input("SibSp (# of siblings / spouses aboard the Titanic)", min_value=0, max_value=10, value=0)


if st.button("Predict"):
    X = np.array([[feature1, feature2, feature3, 1, 1]])
    X_scaled = scaler.transform(X)[:, 0:3]

    prediction = model.predict(X_scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_scaled)[0][1]
        st.write(f"Probability of Surviving: **{prob:.2f}**")

    st.success(f"Prediction: **{prediction}**")
