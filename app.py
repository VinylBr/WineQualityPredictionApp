import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.metrics.plot_confusion_matrix
import matplotlib.pyplot as plt 
plt.style.use(['dark_background'])

st.title("Let's Predict Red Wine Quality")
st.divider()
model_jl_file = "model/wineprediction_model.joblib"

model = load(model_jl_file)

X_red = pd.read_csv("data/red/X_train_df.csv")
y_red = pd.read_csv("data/red/y_train_df.csv")
X_red_test = pd.read_csv("data/red/X_test_df.csv")
y_red_test = pd.read_csv("data/red/y_test_df.csv")


cols_to_transform: ["residual_sugar", "total_sulfur_dioxide"]

features = X_red.columns.values

with st.sidebar:
    st.title("Underlying Properties (Features)")
    sliders = []
    for col in features:
        if col == "density":
            col_slider = st.slider(label = col, min_value = float(X[col].min()), max_value = float(X[col].max()), step = 0.001)#, value = float(X[col].mean()))
        else:    
            col_slider = st.slider(label = col, min_value = float(X[col].min()), max_value = float(X[col].max()))#, value = float(X[col].mean()))
        if col in cols_to_transform:
            col_slider = np.log(col_slider)
            
        sliders.append(col_slider)

#st_scale  = StandardScaler()
#X = st_scale.fit_transform(X)
X_usr = np.array(sliders).reshape(1,-1)

y_pred_svr = model.predict(X_usr)
prediction_prob = model.predict_proba(X_usr)
bestlabelprobability = prediction_prob[(model.classes_ == y_pred_svr).reshape(1,-1)]
st.markdown(f"## Predicted Quality: :red[{y_pred_svr[0]}] (_>5 is Good Wine_)")
st.markdown(f"### :blue[Confidence: {bestlabelprobability[0] :.2f}]")

st.divider()

col1, col2 = st.columns(2)
with col1:
    with st.expander("What is the Feature Importance?"):
        feature_fig, ax = plt.subplots(figsize = (5,4))
        mean_importance = model.feature_importances_ #get feature importance based on minimum decrease in impurity
        sorted_idx = mean_importance.argsort() #sort based on feature importance
        std_importance = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0) #get standard deviation across trees
        ax.barh(pd.Series(features)[sorted_idx],# create horizontal bar plot of sorted feature importance
        mean_importance[sorted_idx], # mean values
        xerr = std_importance, #std
        ecolor = "yellow" #color ofr std wick
        )
        ax.set_xlabel("Perm_importance")
        ax.set_xlim(0, 0.25)
        ax.set_title("Feature Importance") #show the plot
        st.pyplot(feature_fig)

with col2: 
    with st.expander("Model Performance in general"):
        confusion_fig, ax2 = plt.subplots(figsize = (5,4))
        y_pred_red_test = model.predict(X_red_test)
        plot_confusion_matrix(y_red_test, y_pred_red_test, normalize = True)