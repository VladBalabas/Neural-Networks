import pandas as pd
import numpy as np
from keras import Sequential, Input
from matplotlib import pyplot as plt
from tabulate import tabulate

from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

filename = 'output_with_features.csv'
df = pd.read_csv(filename)
choosed_column = df['p']

mean_temp = np.mean(choosed_column)
std_temp = np.std(choosed_column, ddof=1)
normalized_column = (choosed_column - mean_temp) / std_temp

df['p_normalized'] = normalized_column

feature_columns_base = ['p', 'daily_pressure_wave', 'seasonal_pressure_factor']
feature_columns_norm = ['p_normalized', 'daily_pressure_wave', 'seasonal_pressure_factor']

def create_sequences(data, n):
    sequences, targets = [], []
    data = np.array(data)
    for i in range(len(data) - n):
        sequences.append(data[i:i+n])
        targets.append(data[i+n, 0])
    return np.array(sequences), np.array(targets)


train_data, temp_data = train_test_split(df, test_size=0.3, shuffle=False)
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

window_sizes = [2, 4, 8, 12, 28]
neurons_list = [10, 30, 50]
activations = ['relu', 'selu', 'tanh']

window_size = window_sizes[0]
neurons = neurons_list[2]
activation = activations[1]

results = []

data_variants = [
    ('Original', train_data[feature_columns_base], val_data[feature_columns_base], test_data[feature_columns_base]),
    ('Normalized', train_data[feature_columns_norm], val_data[feature_columns_norm], test_data[feature_columns_norm])
]

for data_label, train_set, val_set, test_set in data_variants:
    X_train, y_train = create_sequences(train_set, window_size)
    X_val, y_val = create_sequences(val_set, window_size)
    X_test, y_test = create_sequences(test_set, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(feature_columns_base)))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(feature_columns_base)))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(feature_columns_base)))

    model = Sequential([
        Input(shape=(window_size, len(feature_columns_base))),
        LSTM(neurons, activation=activation, return_sequences=True),
        LSTM(neurons, activation=activation),
        Dropout(0.1),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    if data_label == 'Normalized':
        mse *= std_temp ** 2

    results.append([window_size, neurons, activation, data_label, round(mse, 4)])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve - {data_label}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Actual Values')
    plt.plot(predictions, label='Predictions')
    plt.title(f'Predictions vs Actual - {data_label}')
    plt.legend()
    plt.tight_layout()
    plt.show()

headers = ["Window Size", "Neurons", "Activation", "Data Type", "MSE"]
print(tabulate(results, headers=headers, tablefmt="grid"))

best_model_params = min(results, key=lambda x: x[4])
print("Best model parameters:", best_model_params)


