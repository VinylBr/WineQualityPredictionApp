DB_username = "Vinay Barnabas"
DB_key = "196916"

import numpy as np
import pandas as pd
import streamlit as st
from joblib import dump, load
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
plt.style.use(['dark_background'])

st.title("Let's Predict Red Wine Quality")
st.divider()
model_jl_file = "model/redwine_model.joblib"

model = load(model_jl_file)

X_red = pd.read_csv("data/red/X_train_df.csv")
y_red = pd.read_csv("data/red/y_train_df.csv")
X_red_test = pd.read_csv("data/red/X_test_df.csv")
y_red_test = pd.read_csv("data/red/y_test_df.csv")


cols_to_transform = ["residual_sugar", "total_sulfur_dioxide"] 

features = X_red.columns.values
exp_features = ["Sourness", "Smell", "Refreshing flavor", 
                "Sweetness",
                "Saltiness",
                "Preservative",
                "Thick or heavy it feels",
                "Acidic/basic",
                "Preservatives",
                "Strength of the wine",
                "Molecular Preservatives"]
#targets = [3,4,5,6,7,8]

with st.sidebar:
    st.title("Underlying Properties (Features)")
    sliders = []
    for ind, col in enumerate(features):
        if col == "density":
            col_slider = st.slider(label = col+":"+exp_features[ind], min_value = float(X_red[col].min()), max_value = float(X_red[col].max()), step = 0.001)#, value = float(X[col].mean()))
        else:    
            col_slider = st.slider(label = col+":"+exp_features[ind], min_value = float(X_red[col].min()), max_value = float(X_red[col].max()))#, value = float(X[col].mean()))
        if col in cols_to_transform:
            col_slider = np.log(col_slider)
            
        sliders.append(col_slider)

#st_scale  = StandardScaler()
#X = st_scale.fit_transform(X)
#X_usr = np.array(sliders).reshape(1,-1)
X_usr = pd.DataFrame(np.array(sliders).reshape(1,-1), columns = features)
y_pred_svr = model.predict(X_usr)
prediction_prob = model.predict_proba(X_usr)
bestlabelprobability = prediction_prob[(model.classes_ == y_pred_svr).reshape(1,-1)]
st.markdown(f"## Predicted Quality: :red[{y_pred_svr[0]}] (_>5 is Good Wine_)")
st.markdown(f"### :blue[Confidence: {100*bestlabelprobability[0]:.1f}%]")

st.divider()


with st.expander("Which feature is most importance"):
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
    st.pyplot(feature_fig, use_container_width=True)


with st.expander("Model Performance in general"):
    st.markdown("Numbers = What fraction of the true labels lie in each cell")
    st.write("For example, in row1: for quality of 4(true label), 100% of the times model predicted it to be 5 (predicted label)")
    confusion_fig, ax2 = plt.subplots(figsize = (5,4))
    y_pred_red_test = model.predict(X_red_test)
    
    ConfusionMatrixDisplay.from_predictions(
        y_red_test, y_pred_red_test, xticks_rotation="vertical", normalize = 'true', ax = ax2)
    st.pyplot(confusion_fig, use_container_width=True)
