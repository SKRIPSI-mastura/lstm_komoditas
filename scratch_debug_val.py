import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from preprocessor import prepare_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "commodity_lstm_model.keras")

X, y, classes = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = load_model(MODEL_PATH)
print("--- DEBUG INFO ---")
print("Classes:", list(classes))
print("y_test unique:", np.unique(y_test, return_counts=True))

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Loaded model evaluation on X_test: loss={loss:.4f}, acc={acc:.4f}")

preds = model.predict(X_test[:20], verbose=0)
print("Preds argmax:", np.argmax(preds, axis=1))
print("Preds probabilities sample:\n", preds[:5])
