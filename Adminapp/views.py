from django.shortcuts import render,redirect
from Mainapp.models import*
from Userapp.models import*
from Mainapp.models import User
from Adminapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
import os
import shutil
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from django.contrib import messages
from keras.models import load_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .models import PlainNet, RF, LiteFDNet
import joblib  # Assuming 'RF','DT','XGB is your model in models.py to store the metrics
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from django.shortcuts import render
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6


#gradient boost machine algo for getting acc ,precession , recall , f1 score
# Create your views here.
def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('index')

# def admindashboard(req):
#     return render(req,'admin/admin-dashboard.html')


def admindashboard(req):

    total_users = User.objects.count()

    total_predictions = (
        LiteFDNet.objects.count() +
        PlainNet.objects.count() +
        RF.objects.count()
    )

    latest_model = LiteFDNet.objects.last()
    accuracy = round(latest_model.accuracy * 100, 2) if latest_model else "--"

    context = {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "accuracy": accuracy
    }

    return render(req, 'admin/admin-dashboard.html', context)


import numpy as np
from scipy.stats import kurtosis, skew

def extract_tdf(signal):
    signal = np.asarray(signal, dtype=np.float32)
    rms = np.sqrt(np.mean(signal**2)) + 1e-8

    return [
        np.max(signal),
        np.min(signal),
        np.mean(signal),
        np.mean(np.abs(signal)),
        np.max(signal) - np.min(signal),
        rms,
        np.max(np.abs(signal)) / rms,
        kurtosis(signal),
        np.std(signal),
        skew(signal),
        rms / np.mean(np.abs(signal)),
        rms / np.mean(np.abs(signal)),
        np.var(signal)
    ]

def admingraph(req):
    # Fetch the latest r2_score for each model
    ensemblemodel_details1 = LiteFDNet.objects.last()
    EnsembleModel1 = ensemblemodel_details1.accuracy

    xgb_details2 = PlainNet.objects.last()
    XGB1 = xgb_details2.accuracy

    lr_details2 = RF.objects.last()
    LR1 = lr_details2.accuracy


    print('LiteFDNet','PlainNet','RF')
    print(EnsembleModel1,XGB1,LR1)
    return render(req, 'admin/admin-graph-analysis.html',{'EnsembleModel':EnsembleModel1,'XGB':XGB1,'LR':LR1})

# views.py

def LiteFDNet_btn(request):

    data = pd.read_csv("paderborn_bearing_data.csv")

    raw_signals = data[[f"feature_{i}" for i in range(1024)]].to_numpy(dtype=np.float32)
    y = data["fault_label"].values

    X = np.array([extract_tdf(sig) for sig in raw_signals])

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # model = tf.keras.models.load_model(
    #     "litefdnet_tdf.keras",
    #     compile=False
    # )
    model = tf.keras.models.load_model(
    "litefdnet_tdf.keras",
    custom_objects={"hard_swish": hard_swish},
    compile=False
)

    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    metrics = compute_metrics(y, y_pred)

    LiteFDNet.objects.create(
        name="LiteFDNet (Proposed)",
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"]
    )

    results = LiteFDNet.objects.last()

    context = {
        'name': results.name,
        'accuracy': round(results.accuracy, 2),
        'precision': round(results.precision, 2),
        'recall': round(results.recall, 2),
        'f1_score': round(results.f1_score, 2)
    }
    return render(request, 'admin/LiteFDNet.html', context)

# views.py
def PlainNet_btn(request):

    data = pd.read_csv("paderborn_bearing_data.csv")

    # ===============================
    # 2. Extract RAW signals
    # ===============================
    raw_signals = data[[f"feature_{i}" for i in range(1024)]].to_numpy(dtype=np.float32)
    y = data["fault_label"].values

    # ===============================
    # 3. Convert RAW → TDF
    # ===============================
    X = np.array([extract_tdf(sig) for sig in raw_signals])

    # ===============================
    # 4. Normalize (same as training)
    # ===============================
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # ===============================
    # 5. Load pretrained PlainNet
    # ===============================
    model = tf.keras.models.load_model(
        "PlainNet_pretrained1.keras",
        compile=False
    )

    # ===============================
    # 6. Predict
    # ===============================
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)

    # ===============================
    # 7. Compute metrics
    # ===============================
    metrics = compute_metrics(y, y_pred)

    # ===============================
    # 8. Save results to DB
    # ===============================
    PlainNet.objects.create(
        name="PlainNet (Baseline DL)",
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"]
    )

    results = PlainNet.objects.last()

    # ===============================
    # 9. Context (YOUR FORMAT)
    # ===============================
    context = {
        'name': results.name,
        'accuracy': round(results.accuracy, 2),
        'precision': round(results.precision, 2),
        'recall': round(results.recall, 2),
        'f1_score': round(results.f1_score, 2)
    }

    return render(request, 'admin/PlainNet.html', context)

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def RF_btn(request):

    data = pd.read_csv("paderborn_bearing_data.csv")

     # ===== Extract RAW signals =====
    raw_signals = data[[f"feature_{i}" for i in range(1024)]].to_numpy(dtype=np.float32)
    y = data["fault_label"].values

    # ===== Convert RAW → TDF =====
    X = np.array([extract_tdf(sig) for sig in raw_signals])

    # ===== Load pretrained RF =====
    model = joblib.load("RF_pretrained1.pkl")

    # ===== Predict =====
    y_pred = model.predict(X)

    # ===== Metrics =====
    metrics = compute_metrics(y, y_pred)

    RF.objects.create(
        name="Random Forest (Baseline ML)",
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1_score=metrics["f1_score"]
    )

    results = RF.objects.last()

    context = {
        'name': results.name,
        'accuracy': round(results.accuracy, 2),
        'precision': round(results.precision, 2),
        'recall': round(results.recall, 2),
        'f1_score': round(results.f1_score, 2)
    }

    return render(request, 'admin/RF.html', context)