{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPtB/Sfy1tS9mGujKNetRb0",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/VinylBr/WineQualityPredictionApp/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.svm import SVR"
      ],
      "metadata": {
        "id": "EbRU8at5Q39B"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "USJJKWi8QQJC"
      },
      "outputs": [],
      "source": [
        "C = 0.55 #GridSearchCV\n",
        "epsilon = 0.255 #from GridSearchCV\n",
        "best_svr = SVR(C = C, epsilon = epsilon)\n",
        "best_svr.fit(X_total_train_red, y_total_train_red)\n",
        "y_pred_svr = best_svr.predict(X_final_test_red)\n",
        "mse_svr = mean_squared_error(y_final_test_red, y_pred_svr)\n",
        "print(f\"MSE on the test set with C: {C: .2f} and epsilon: {epsilon: .2f} is {mse_svr: .2f}\")"
      ]
    }
  ]
}