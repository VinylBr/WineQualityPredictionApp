import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 

st.title("Wine Quality")
st.write("Hi! What factors control wine quality? Let's find out")
#model_jl_file = "model/wineprediction_model.joblib"

#model = load(model_jl_file)

X = pd.read_csv("data/X_df.csv")
y = pd.read_csv("data/y_df.csv")

max_depth = 21
n_estimators = 41


tab1, tab2 = st.tabs(["Prediction", "Feature Importance"]) #tab1 for prediction and tab2 for feature importance

cols_to_transform = ["residual_sugar", "free_sulfur_dioxide", "Bound_sulfur_dioxide"]

features = X.columns.values


sliders = []
for col in features:
    col_slider = st.slider(label = col, min_value = float(X[col].min()), max_value = float(X[col].max()), value = float(X[col].mean()))
    if col in cols_to_transform:
        col_slider = np.log(col_slider)
    sliders.append(col_slider)

st_scale  = StandardScaler()
X = st_scale.fit_transform(X)
X_test = np.array(sliders).reshape(1,-1)
best_forest_class = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)
best_forest_class.fit(X, y['quality'])
scaled_test_features = st_scale.transform(X_test)

y_pred_svr = best_forest_class.predict(X_test)
