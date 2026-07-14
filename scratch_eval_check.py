import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from preprocessor import prepare_data
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "commodity_lstm_model.keras")

X, y, classes = prepare_data()
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model(MODEL_PATH)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("=== MODEL EVALUATION CHECK ===")
print("Loaded model evaluate accuracy:", acc)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
print("Scikit-learn accuracy score:", accuracy_score(y_test, y_pred))
