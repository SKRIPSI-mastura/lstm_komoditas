from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
)

def create_lstm_model(input_shape, num_classes):
    """
    Mendefinisikan arsitektur model LSTM untuk klasifikasi komoditas.
    Model v1: single-input Sequential.
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


def create_lstm_model_v2(seq_input_shape, n_static_features, num_classes):
    """
    Model LSTM v2: Dual-Input Hybrid Architecture.

    Branch 1 — Sequential (LSTM):
        Input: (SEQ_LENGTH, 3) — fitur iklim harian (T2M, RH2M, WS2M)
        Layers: LSTM(128) → Dropout → LSTM(64) → Dropout

    Branch 2 — Static (Dense):
        Input: (n_static_features,) — fitur lahan statis
                (elevasi, curah_hujan, pH, liat, pasir, debu)
        Layers: Dense(32, relu) → BatchNorm → Dense(16, relu)

    Merged:
        Concatenate → Dense(64, relu) → Dropout → Dense(32, relu) → Softmax(num_classes)

    Args:
        seq_input_shape   : tuple (SEQ_LENGTH, n_seq_features), misal (30, 3)
        n_static_features : int, jumlah fitur statis, misal 6
        num_classes       : int, jumlah kelas komoditas, misal 7
    """
    # --- Branch 1: Sequential LSTM ---
    input_seq = Input(shape=seq_input_shape, name='input_sequential')
    x = LSTM(128, return_sequences=True, activation='tanh')(input_seq)
    x = Dropout(0.2)(x)
    x = LSTM(64, activation='tanh')(x)
    x = Dropout(0.2)(x)

    # --- Branch 2: Static Dense ---
    input_static = Input(shape=(n_static_features,), name='input_static')
    s = Dense(32, activation='relu')(input_static)
    s = BatchNormalization()(s)
    s = Dense(16, activation='relu')(s)

    # --- Merge ---
    merged = Concatenate()([x, s])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(32, activation='relu')(merged)
    output = Dense(num_classes, activation='softmax', name='output')(merged)

    model = Model(inputs=[input_seq, input_static], outputs=output,
                  name='LSTM_v2_HybridDualInput')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
