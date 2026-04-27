from src.preprocess import load_data, clean_data, feature_engineering, encode_data, split_data
from src.train import train_models
from src.evaluate import evaluate_model

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os

TRAIN_PATH = "data/fraudTrain.csv"
TEST_PATH = "data/fraudTest.csv"

# -------------------------------
# 1. Load
# -------------------------------
train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

# -------------------------------
# 2. Clean
# -------------------------------
train_df = clean_data(train_df)
test_df = clean_data(test_df)

# -------------------------------
# SAVE DROPDOWN OPTIONS (for UI)
# -------------------------------
options = {
    "category": train_df["category"].astype(str).unique().tolist(),
    "merchant": train_df["merchant"].astype(str).unique().tolist(),
    "city": train_df["city"].astype(str).unique().tolist(),
    "state": train_df["state"].astype(str).unique().tolist(),
    "job": train_df["job"].astype(str).unique().tolist()
}
# -------------------------------
# 3. Feature Engineering
# -------------------------------
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# -------------------------------
# CORRECT STATE → CITY MAP
# -------------------------------
state_city_map = (
    train_df.groupby("state")["city"]
    .unique()
    .apply(list)
    .to_dict()
)

joblib.dump(state_city_map, "models/state_city_map.pkl")

# -------------------------------
# 4. Encode
# -------------------------------
train_df, test_df, encoder = encode_data(train_df, test_df)

# -------------------------------
# 5. Split
# -------------------------------
X_train, y_train = split_data(train_df)
X_test, y_test = split_data(test_df)

# -------------------------------
# 6. Handle imbalance
# -------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -------------------------------
# 7. Train models
# -------------------------------
models = train_models(X_train_res, y_train_res)

# -------------------------------
# 8. Evaluate & Select Best Model
# -------------------------------
best_model = None
best_score = 0

print("\n📊 Model Performance (baseline threshold=0.5):\n")

for name, model in models.items():
    report, roc_auc = evaluate_model(model, X_test, y_test, threshold=0.5)

    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']

    print(f"===== {name} =====")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model
        best_name = name

# -------------------------------
# 9. Auto Threshold Selection
# -------------------------------
print("\n🔍 Auto Threshold Tuning:\n")

y_prob = best_model.predict_proba(X_test)[:, 1]

best_threshold = 0.5
best_f1 = 0

thresholds = np.arange(0.3, 0.8, 0.05)

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred)

    print(f"Threshold: {t:.2f} → F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"\n🏆 Best Threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

# -------------------------------
# 10. Final Evaluation (best threshold)
# -------------------------------
report, roc_auc = evaluate_model(best_model, X_test, y_test, threshold=best_threshold)

precision = report['1']['precision']
recall = report['1']['recall']
f1 = report['1']['f1-score']

print("\n📊 Final Model Performance:\n")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

# -------------------------------
# 11. Confusion Matrix
# -------------------------------
y_pred = (y_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

# -------------------------------
# 12. Feature Importance (RF only)
# -------------------------------
if best_name == "Random Forest":
    importances = best_model.feature_importances_
    feature_names = X_train.columns

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12,6))
    plt.title("Feature Importance")

    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), feature_names[indices], rotation=90)

    plt.tight_layout()
    plt.show()
    
joblib.dump(X_train.columns.tolist(), "models/features.pkl")
# -------------------------------
# 13. Save model + encoder
# -------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/model.pkl")
joblib.dump(encoder, "models/encoder.pkl")
joblib.dump(options, "models/options.pkl")

print(f"\n🏆 Best Model: {best_name}")
print("✅ Model + Encoder saved successfully!")