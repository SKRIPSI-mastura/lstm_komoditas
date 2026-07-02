from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def create_lstm_model(input_shape, num_classes):
    """
    Mendefinisikan arsitektur model LSTM untuk klasifikasi komoditas.
    """
    model = Sequential([
        Input(shape=input_shape),
        # Layer LSTM Pertama
        LSTM(128, return_sequences=True, activation='tanh'),
        Dropout(0.2),
        # Layer LSTM Kedua
        LSTM(64, activation='tanh'),
        Dropout(0.2),
        # Fully Connected Layers
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        # Output Layer (Softmax untuk klasifikasi multi-kelas)
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
