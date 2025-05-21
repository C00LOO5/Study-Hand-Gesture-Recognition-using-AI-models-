import pandas as pd
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model

svm = joblib.load("svm_model.pkl")
knn = joblib.load("knn_model.pkl")
rf = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

cnn = load_model("cnn_model.keras")
lstm = load_model("lstm_model.keras")
transformer = load_model("transformer_model.keras")

test_ml_df = pd.read_csv("test.csv")
X_test_ml = test_ml_df.drop(columns=["label"])
y_test_ml = test_ml_df["label"]
y_test_ml_encoded = label_encoder.transform(y_test_ml)

X_test_ml_scaled = scaler.transform(X_test_ml)

pred_svm = svm.predict(X_test_ml_scaled)
pred_knn = knn.predict(X_test_ml_scaled)
pred_rf = rf.predict(X_test_ml_scaled)

def evaluate_model(y_true, y_pred, model_name, results_dict):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1) * 100
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1) * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100
    results_dict[model_name] = [accuracy, precision, recall, f1]
    print(f"{model_name} - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-score: {f1:.2f}%")

results = {}
print("Evaluating ML models...")
evaluate_model(y_test_ml_encoded, pred_svm, "SVM", results)
evaluate_model(y_test_ml_encoded, pred_knn, "KNN", results)
evaluate_model(y_test_ml_encoded, pred_rf, "Random Forest", results)

with open("test.json", "r") as f:
    test_dl_data = json.load(f)

X_test_dl = []
y_test_dl = []

for entry in test_dl_data:
    X_test_dl.append(entry["hand_landmarks"])
    y_test_dl.append(entry["label"])

X_test_dl = np.array(X_test_dl)
y_test_dl_encoded = label_encoder.transform(y_test_dl)

X_test_dl = X_test_dl.reshape(X_test_dl.shape[0], 21, 2)
y_test_dl_one_hot = np.eye(len(label_encoder.classes_))[y_test_dl_encoded]

pred_cnn = np.argmax(cnn.predict(X_test_dl), axis=1)
pred_lstm = np.argmax(lstm.predict(X_test_dl), axis=1)
pred_transformer = np.argmax(transformer.predict(X_test_dl), axis=1)

print("\nEvaluating DL models...")
evaluate_model(y_test_dl_encoded, pred_cnn, "CNN", results)
evaluate_model(y_test_dl_encoded, pred_lstm, "LSTM", results)
evaluate_model(y_test_dl_encoded, pred_transformer, "Transformer", results)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    misclassified = np.argsort(np.sum(cm, axis=1) - np.diag(cm))[-5:]  # Top 5 misclassified
    cm_filtered = cm[misclassified][:, misclassified]
    labels = label_encoder.classes_[misclassified]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_filtered, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name} (Top Misclassified)")
    plt.show()

plot_confusion_matrix(y_test_ml_encoded, pred_svm, "SVM")
plot_confusion_matrix(y_test_ml_encoded, pred_knn, "KNN")
plot_confusion_matrix(y_test_ml_encoded, pred_rf, "Random Forest")
plot_confusion_matrix(y_test_dl_encoded, pred_cnn, "CNN")
plot_confusion_matrix(y_test_dl_encoded, pred_lstm, "LSTM")
plot_confusion_matrix(y_test_dl_encoded, pred_transformer, "Transformer")
