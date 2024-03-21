DB_username = "Vinay Barnabas"
DB_key = "196916"

import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt 
plt.style.use(['dark_background'])

st.title("Let's Predict Red Wine Quality")
st.divider()
model_jl_file = "model/redwine_model.joblib"

model_red = load(model_jl_file)

X_red = pd.read_csv("data/red/X_train_df.csv")
y_red = pd.read_csv("data/red/y_train_df.csv")
X_red_test = pd.read_csv("data/red/X_test_df.csv")
y_red_test = pd.read_csv("data/red/y_test_df.csv")


cols_to_transform = ["residual_sugar", "total_sulfur_dioxide"] 

features = X_red.columns.values
exp_features = ["Sourness: Fixed acidity (g/L)", 
                "Smell: Volatile acidity (g/L)", 
                "Citrus: Citric acid (g/L)", 
                "Sweetness: Sugar (g/L)",
                "Saltiness: Chlorides (g/L)",
                "Preservative: Total SO2 (ppm)",
                "Heavy or light: Density (g/L)",
                "Acidic/Basic: pH",
                "Preservatives: Sulphates (ppm)",
                "Alcohol Content (%vol)",
                "Preservatives: Molecular SO2 (ppm)"]
#feature_merge = np.concatenate(features, exp_features, axis = 2)
feature_merge = [i + ":" + j for i, j in zip(features, exp_features)]
with st.sidebar:
    st.title("Underlying Properties (Features)")
    sliders = []
    for ind, col in enumerate(features):
        if col == "density":
            col_slider = st.slider(label = exp_features[ind], min_value = float(X_red[col].min()), max_value = float(X_red[col].max()), step = 0.001)#, value = float(X[col].mean()))
            #st.markdown(f"*{exp_features[ind]}*: ")
            #st.divider()
        else:    
            col_slider = st.slider(label = exp_features[ind], min_value = float(X_red[col].min()), max_value = float(X_red[col].max()))#, value = float(X[col].mean()))
            #st.markdown(f"*{exp_features[ind]}*: ")
            #st.divider()
        if col in cols_to_transform:
            col_slider = np.log(col_slider)
            #st.divider()
        sliders.append(col_slider)

#st_scale  = StandardScaler()
#X = st_scale.fit_transform(X)
#X_usr = np.array(sliders).reshape(1,-1)
X_usr = pd.DataFrame(np.array(sliders).reshape(1,-1), columns = features)
y_pred_svr = model_red.predict(X_usr)
prediction_prob = model_red.predict_proba(X_usr)
bestlabelprobability = prediction_prob[(model_red.classes_ == y_pred_svr).reshape(1,-1)]
st.markdown(f"## Predicted Quality: :red[{y_pred_svr[0]}] (_>5 is Good Wine_)")
st.markdown(f"### :blue[Confidence: {100*bestlabelprobability[0]:.1f}%]")

st.divider()


with st.expander("Which feature is most importance"):
    feature_fig, ax = plt.subplots(figsize = (5,4))
    n_repeats = 11
    feature_importance = permutation_importance(model_red, X_red_test, y_red_test, random_state = 11, n_repeats = n_repeats) #feature importance for RandomForest

    mean_importance = feature_importance.importances_mean #get feature importance based on minimum decrease in impurity
    sorted_idx = mean_importance.argsort() #sort based on feature importance
    std_importance = feature_importance.importances_std #get standard deviation across trees
    ax.barh(pd.Series(features)[sorted_idx],# create horizontal bar plot of sorted feature importance
    mean_importance[sorted_idx], # mean values
    xerr = std_importance, #std
    ecolor = "yellow" #color ofr std wick
    )
    ax.set_xlabel("Importance measure")
    ax.set_xlim(0, 0.15)
    ax.set_title("Feature Importance") #show the plot
    st.pyplot(feature_fig, use_container_width=True)


with st.expander("Model Performance"):
    st.markdown("How confused is the ML model :sweat_smile:")
    st.write("For example, in row1: for quality of 4(true label), 100% of the times model predicted it to be 5 (predicted label), *Pretty confused with quality 4*")
    confusion_fig, ax2 = plt.subplots(figsize = (5,4))
    y_pred_red_test = model_red.predict(X_red_test)
    
    ConfusionMatrixDisplay.from_predictions(
        y_red_test, y_pred_red_test, xticks_rotation="vertical", normalize = 'true', ax = ax2)
    st.pyplot(confusion_fig, use_container_width=True)
