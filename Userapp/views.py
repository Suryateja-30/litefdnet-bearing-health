from django.shortcuts import render,redirect
from Mainapp.models import *
from Userapp.models import Dataset
from Adminapp.models import *
from django.contrib import messages
import time
from django.core.paginator import Paginator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.shortcuts import render, redirect
from .models import User  # Ensure your models are imported
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def userdashboard(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    context = {
        "la": user
    }
    return render(req,'user/user-dashboard.html',context)
  



def userlogout(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(req, 'You are logged out..')
    return redirect('index')









# def LiteFDNet_Predict_Form_btn(request):

#     import numpy as np
#     import tensorflow as tf
#     from sklearn.preprocessing import MinMaxScaler

#     prediction = None
#     confidence = None
#     explanation = {}

#     if request.method == "POST":

#         # ===============================
#         # 1. Read form inputs
#         # ===============================
#         rms = float(request.POST.get("rms"))
#         kurt = float(request.POST.get("kurtosis"))
#         skew = float(request.POST.get("skewness"))
#         p2p = float(request.POST.get("p2p"))
#         crest = float(request.POST.get("crest"))

#         # ===============================
#         # 2. Build TDF vector (13 features)
#         #    Order MUST match training
#         # ===============================
#         tdf = [
#             p2p,                 # max-min approx
#             -p2p,                # min approx
#             rms,                 # mean approx
#             abs(rms),            # abs mean
#             p2p,                 # peak-to-peak
#             rms,                 # RMS
#             crest,               # crest factor
#             kurt,                # kurtosis
#             rms * 0.1,           # std approx
#             skew,                # skewness
#             crest * 0.5,         # form factor approx
#             crest * 0.5,         # shape factor approx
#             rms ** 2             # variance
#         ]

#         # Convert to NumPy
#         X_np = np.array([tdf], dtype=np.float32)

#         # ===============================
#         # 3. Normalize (same logic as training)
#         # ===============================
#         scaler = MinMaxScaler()
#         X_np = scaler.fit_transform(X_np)

#         # ✅ Convert NumPy → Tensor (CRITICAL FIX)
#         X = tf.convert_to_tensor(X_np, dtype=tf.float32)

#         # ===============================
#         # 4. Load LiteFDNet
#         # ===============================
#         model = tf.keras.models.load_model(
#             "litefdnet_tdf.keras",
#             compile=False
#         )

#         # ===============================
#         # 5. Prediction
#         # ===============================
#         probs = model(X, training=False)
#         pred_class = int(tf.argmax(probs, axis=1).numpy()[0])
#         confidence = float(tf.reduce_max(probs).numpy())

#         class_map = {
#             0: "Healthy",
#             1: "Inner Ring Fault",
#             2: "Outer Ring Fault"
#         }
#         prediction = class_map[pred_class]

#         # ===============================
#         # 6. XAI – Gradient-based Explanation
#         # ===============================
#         with tf.GradientTape() as tape:
#             tape.watch(X)
#             preds = model(X, training=False)
#             target = preds[:, pred_class]

#         grads = tape.gradient(target, X).numpy()[0]

#         feature_names = [
#             "RMS",
#             "Kurtosis",
#             "Skewness",
#             "Peak-to-Peak",
#             "Crest Factor"
#         ]

#         # Use only top 5 feature gradients
#         grad_scores = np.abs(grads[:5])

#         explanation = {
#             fname: round(float(score), 4)
#             for fname, score in zip(feature_names, grad_scores)
#         }

#     context = {
#         "prediction": prediction,
#         "confidence": round(confidence, 3) if confidence else None,
#         "explanation": explanation
#     }

#     return render(request, "user/litefdnet_predict.html", context)

def LiteFDNet_Predict_Form_btn(request):

    import numpy as np
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    import shap
    import lime
    import lime.lime_tabular

    prediction = None
    confidence = None
    gradient_explanation = {}
    shap_explanation = {}
    lime_explanation = {}

    # -------------------------------
    # Feature names (TOP TDF features)
    # -------------------------------
    feature_names = [
        "RMS",
        "Kurtosis",
        "Skewness",
        "Peak-to-Peak",
        "Crest Factor"
    ]

    if request.method == "POST":

        # ===============================
        # 1. Read form inputs
        # ===============================
        rms = float(request.POST.get("rms"))
        kurt = float(request.POST.get("kurtosis"))
        skew = float(request.POST.get("skewness"))
        p2p = float(request.POST.get("p2p"))
        crest = float(request.POST.get("crest"))

        # ===============================
        # 2. Build 13-D TDF vector
        #    (MUST match training order)
        # ===============================
        tdf = [
            p2p,                 # max-min approx
            -p2p,                # min approx
            rms,                 # mean approx
            abs(rms),            # abs mean
            p2p,                 # peak-to-peak
            rms,                 # RMS
            crest,               # crest factor
            kurt,                # kurtosis
            rms * 0.1,           # std approx
            skew,                # skewness
            crest * 0.5,         # form factor
            crest * 0.5,         # shape factor
            rms ** 2             # variance
        ]

        # ===============================
        # 3. Normalize
        # ===============================
        X_np = np.array([tdf], dtype=np.float32)
        scaler = MinMaxScaler()
        X_np = scaler.fit_transform(X_np)

        # Tensor for TF
        X = tf.convert_to_tensor(X_np, dtype=tf.float32)

        # ===============================
        # 4. Load LiteFDNet
        # ===============================
        model = tf.keras.models.load_model(
            "litefdnet_tdf.keras",
            compile=False
        )

        # ===============================
        # 5. Prediction
        # ===============================
        probs = model(X, training=False)
        pred_class = int(tf.argmax(probs, axis=1).numpy()[0])
        confidence = float(tf.reduce_max(probs).numpy())

        class_map = {
            0: "Healthy",
            1: "Inner Ring Fault",
            2: "Outer Ring Fault"
        }
        prediction = class_map[pred_class]

        # =====================================================
        # 6. Gradient-based XAI (already working – kept)
        # =====================================================
        with tf.GradientTape() as tape:
            tape.watch(X)
            preds = model(X, training=False)
            target = preds[:, pred_class]

        grads = tape.gradient(target, X).numpy()[0]
        grad_scores = np.abs(grads[:5])

        gradient_explanation = {
            fname: round(float(score), 4)
            for fname, score in zip(feature_names, grad_scores)
        }

        # =====================================================
        # 7. SHAP Explanation (KernelExplainer – safe)
        # =====================================================
        # =====================================================
        # 7. SHAP Explanation (FIXED & SAFE)
        # =====================================================
        background = np.zeros((10, X_np.shape[1]), dtype=np.float32)

        def model_predict(x):
            return model.predict(x, verbose=0)

        shap_explainer = shap.KernelExplainer(model_predict, background)
        shap_values = shap_explainer.shap_values(X_np, nsamples=100)

        # SHAP vector for predicted class
        shap_vector = np.abs(shap_values[pred_class][0])

        # Number of features we can safely explain
        num_feats = min(len(feature_names), len(shap_vector))

        shap_explanation = {
            feature_names[i]: round(float(shap_vector[i]), 4)
            for i in range(num_feats)
}

       # =====================================================
# 8. LIME Explanation (FIXED & ROBUST)
# =====================================================
    try:
        # Create a non-degenerate background (IMPORTANT)
        lime_background = X_np + np.random.normal(
            loc=0.0,
            scale=0.01,
            size=(50, X_np.shape[1])
        )

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=lime_background,
            feature_names=feature_names + [f"f{i}" for i in range(8)],
            class_names=["Healthy", "IR Fault", "OR Fault"],
            mode="classification",
            discretize_continuous=False
        )

        lime_exp = lime_explainer.explain_instance(
            X_np[0],
            model_predict,
            num_features=min(5, X_np.shape[1])
        )

        lime_explanation = {
            feature: round(weight, 4)
            for feature, weight in lime_exp.as_list()
        }

    except Exception as e:
        # Failsafe for web deployment
        lime_explanation = {
            "LIME Error": "Local explanation not available for this input"
        }
    # ===============================
    # 9. Context (Template-ready)
    # ===============================
    context = {
        "prediction": prediction,
        "confidence": round(confidence, 3) if confidence else None,
        "gradient_explanation": gradient_explanation,
        "shap_explanation": shap_explanation,
        "lime_explanation": lime_explanation
    }

    return render(request, "user/litefdnet_predict.html", context)













        





    
   

