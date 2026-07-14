import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from preprocessor import prepare_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "commodity_lstm_model.keras")

X, y, classes = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sub-sample 25% train
np.random.seed(42)
indices = np.random.choice(len(X_train), size=int(len(X_train) * 0.25), replace=False)
X_train_sub = X_train[indices]
y_train_sub = y_train[indices]

model = load_model(MODEL_PATH)
loss_train, acc_train = model.evaluate(X_train_sub, y_train_sub, batch_size=256, verbose=0)
loss_test, acc_test = model.evaluate(X_test, y_test, batch_size=256, verbose=0)

print("=== CHECK TRAIN VS TEST ACCURACY ===")
print("Train (sub-sampled 25%) accuracy:", acc_train)
print("Test (full 20% validation) accuracy:", acc_test)
