import numpy as np
import pandas as pd
import joblib
import threading
from playsound import playsound
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
import sqlite3
from datetime import timedelta

# ✅ Load trained models
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
svm_model = joblib.load("svm_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
ann_model = joblib.load("ann_model.pkl")

# ✅ Initialize Flask App
app = Flask(__name__)
app.secret_key = "my_secure_secret"
app.permanent_session_lifetime = timedelta(minutes=30)

# ✅ Alarm Sound Function
def play_alarm():
    playsound("alarm.mp3")

# ✅ Database Connection
def get_db_connection():
    conn = sqlite3.connect("admin.db")
    conn.row_factory = sqlite3.Row
    return conn

# ✅ Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM admin_users WHERE username = ? AND password = ?", 
                            (username, password)).fetchone()
        conn.close()

        if user:
            session["admin"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="❌ Invalid username or password.")
    return render_template("login.html")

# ✅ Logout Route
@app.route("/logout")
def logout():
    session.pop("admin", None)
    flash("🔒 Logged out successfully.")
    return redirect(url_for("login"))

# ✅ Home Route
@app.route("/")
@app.route("/home")
def home():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ✅ About Route
@app.route("/about")
def about():
    if "admin" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")

# ✅ Predict Route
@app.route("/predict", methods=["POST"])
def predict():
    if "admin" not in session:
        return redirect(url_for("login"))

    try:
        input_data = [float(x) for x in request.form.values()]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        rf_pred = rf_model.predict(input_scaled)[0]
        dt_pred = dt_model.predict(input_scaled)[0]
        svm_pred = svm_model.predict(input_scaled)[0]
        xgb_pred = xgb_model.predict(input_scaled)[0]
        ann_pred = (ann_model.predict(input_scaled) > 0.5).astype(int)[0][0]

        predictions = {
            "Random Forest Prediction": "Theft" if rf_pred == 1 else "Normal",
            "Decision Tree Prediction": "Theft" if dt_pred == 1 else "Normal",
            "SVM Prediction": "Theft" if svm_pred == 1 else "Normal",
            "XGBoost Prediction": "Theft" if xgb_pred == 1 else "Normal",
            "ANN Prediction": "Theft" if ann_pred == 1 else "Normal"
        }

        theft_count = list(predictions.values()).count("Theft")
        final_decision = "Theft" if theft_count >= 3 else "Normal"

        if final_decision == "Theft":
            threading.Thread(target=play_alarm).start()

        return render_template("index.html", prediction=predictions, final_decision=final_decision)

    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ Run App
if __name__ == "__main__":
    app.run(debug=True)