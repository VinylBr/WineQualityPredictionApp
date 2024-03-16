import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

st.write("
# Hello! How does the quality of wine change with composition?
")



user_data = 

C = 0.55 #GridSearchCV
epsilon = 0.255 #from GridSearchCV
best_svr = SVR(C = C, epsilon = epsilon)
best_svr.fit(X_total_train_red, y_total_train_red)
y_pred_svr = best_svr.predict(user_data)