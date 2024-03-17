import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 

st.title("Wine Quality")
st.write("Hi! What factors control wine quality? Let's find out")
#model_jl_file = "model/wineprediction_model.joblib"

#model = load(model_jl_file)

X = pd.read_csv("data/df_X.csv")
y = pd.read_csv("data/df_y.csv")
max_depth = 21
n_estimators = 61


tab1, tab2 = st.tabs #tab1 for prediction and tab2 for feature importance

cols_to_transform = ["residual_sugar", "free_sulfur_dioxide", "Bound_sulfur_dioxide"]

st_scale  = StandardScaler()
X = st_scale.fit_transform(features_df)

best_forest_class = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)
best_forest_class.fit(X_process_train_mouse, y_process_train_mouse)

sliders = []
for col in features_df.columns:
    col_slider = st.slider(label = col, min_value = float(features_df[col].min()), max_value = float(features_df[col].max()), value = float(features_df[col].mean()))
    if col in cols_to_transform:
        col_slider = np.log(col_slider)
    sliders.append(col_slider)

scaled_test_features = st_scale.transform(sliders)

y_pred_svr = model.predict(sliders)
