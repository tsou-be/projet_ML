import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------
# entrainer le modèle
# -----------------------
train_data = pd.read_csv("ex2data1.txt", header=None)

X_train = train_data[[0,1]]
y_train = train_data[2]

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------
# interface
# -----------------------
st.title("Entrez vos données")
uploaded_file = st.file_uploader(
    "Importer fichier",
    type=["csv","xlsx","txt"]
)

if uploaded_file is not None:

    # lire fichier
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)

    else:
        data = pd.read_csv(uploaded_file)

    st.write("Données importées")
    st.dataframe(data)

    
    exam1_col = st.selectbox("Colonne Exam 1", data.columns)
    exam2_col = st.selectbox("Colonne Exam 2", data.columns)

    if st.button("Prédire"):

        X_new = data[[exam1_col, exam2_col]]

        predictions = model.predict(X_new)

        data["Prediction"] = predictions

        st.write("Résultats")
        st.dataframe(data)