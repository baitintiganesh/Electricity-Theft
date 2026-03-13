import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU

# ✅ Load the preprocessed dataset
df = pd.read_csv("electricity_theft_preprocessed.csv")  # Ensure the file is in the same directory

# ✅ Split features and labels
X = df.drop(columns=['Theft_Label'])  # Features (5 columns retained)
y = df['Theft_Label']  # Target variable (0 = Normal, 1 = Theft)

# ✅ Split into training & testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Standardize data (important for SVM & Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ Save the updated scaler for Flask app
joblib.dump(scaler, "scaler.pkl")

print(f"✅ All Features Kept: {X.shape[1]} features")  # Verify 5 features are retained

# ✅ Dictionary to store model results
model_results = {}

# -----------------------------------------
# 1️⃣ Decision Tree Model
from sklearn.model_selection import GridSearchCV

dt_params = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
dt_grid.fit(X_train, y_train)

best_dt = dt_grid.best_estimator_
y_pred_dt_tuned = best_dt.predict(X_test)

model_results["Decision Tree (Tuned)"] = accuracy_score(y_test, y_pred_dt_tuned)
joblib.dump(best_dt, "decision_tree_model.pkl")  # Save model

print(f"✅ Best Decision Tree Model: {dt_grid.best_params_}")
print(f"🎯 Tuned Decision Tree Accuracy: {model_results['Decision Tree (Tuned)']:.4f}")

# -----------------------------------------
# 2️⃣ Random Forest Model
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)
model_results["Random Forest (Tuned)"] = accuracy_score(y_test, y_pred_rf_tuned)
joblib.dump(best_rf, "random_forest_model.pkl")  # Save model

print(f"✅ Best Random Forest Model: {rf_grid.best_params_}")
print(f"🎯 Tuned Random Forest Accuracy: {model_results['Random Forest (Tuned)']:.4f}")

# -----------------------------------------
# 3️⃣ Support Vector Machine (SVM)
svm_params = {
    'C': [0.1, 1, 10, 50, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
svm_grid.fit(X_train, y_train)

best_svm = svm_grid.best_estimator_
y_pred_svm_tuned = best_svm.predict(X_test)

model_results["SVM (Improved)"] = accuracy_score(y_test, y_pred_svm_tuned)
joblib.dump(best_svm, "svm_model.pkl")  # Save model

print(f"✅ Best SVM Model: {svm_grid.best_params_}")
print(f"🎯 Improved SVM Accuracy: {model_results['SVM (Improved)']:.4f}")

# -----------------------------------------
# 4️⃣ XGBoost Model
from sklearn.model_selection import RandomizedSearchCV

xgb_params = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 6, 9, 12],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_search = RandomizedSearchCV(xgb.XGBClassifier(random_state=42), xgb_params, cv=3, scoring='accuracy', n_iter=20, n_jobs=-1, verbose=2)
xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_
y_pred_xgb_tuned = best_xgb.predict(X_test)
model_results["XGBoost (Tuned)"] = accuracy_score(y_test, y_pred_xgb_tuned)
joblib.dump(best_xgb, "xgboost_model.pkl")  # Save model

print(f"✅ Best XGBoost Model: {xgb_search.best_params_}")
print(f"🎯 Tuned XGBoost Accuracy: {model_results['XGBoost (Tuned)']:.4f}")

# -----------------------------------------
# 5️⃣ Artificial Neural Network (ANN)
ann_model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.01),
    Dropout(0.4),

    Dense(128),
    LeakyReLU(alpha=0.01),
    Dropout(0.3),

    Dense(64),
    LeakyReLU(alpha=0.01),
    Dropout(0.3),

    Dense(32),
    LeakyReLU(alpha=0.01),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                  loss='binary_crossentropy', metrics=['accuracy'])

ann_model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=1, validation_data=(X_test, y_test))

y_pred_ann_tuned = (ann_model.predict(X_test) > 0.5).astype(int).flatten()
model_results["ANN (Improved)"] = accuracy_score(y_test, y_pred_ann_tuned)
joblib.dump(ann_model, "ann_model.pkl")  # Save ANN model

print(f"🎯 Improved ANN Accuracy: {model_results['ANN (Improved)']:.4f}")

# -----------------------------------------
# Print Model Results
print("\n Model Accuracy Results:")
for model, accuracy in model_results.items():
    print(f"{model}: {accuracy:.4f}")

# Print Classification Report for Best Model (Random Forest / XGBoost recommended)
print("\n Classification Report (Best Model - XGBoost):")
print(classification_report(y_test, y_pred_xgb_tuned))
