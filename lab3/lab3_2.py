import pandas as pd
import numpy as np
from keras import Sequential, Input
from matplotlib import pyplot as plt
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('output.csv')
choosed_column = df['p']

mean_temp = np.mean(choosed_column)
std_temp = np.std(choosed_column, ddof=1)
normalized_column = (choosed_column - mean_temp) / std_temp

df['p_normalized'] = normalized_column


def create_sequences(data, n):
    sequences, targets = [], []
    data = np.array(data)
    for i in range(len(data) - n):
        sequences.append(data[i:i + n])
        targets.append(data[i + n])
    return np.array(sequences), np.array(targets)


train_data, temp_data = train_test_split(df, test_size=0.3, shuffle=False)
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

window_sizes = [2, 4, 8, 12, 28]
neurons_list = [10, 30, 50]
activations = ['relu', 'selu', 'tanh']

window_size = window_sizes[0]
neurons = neurons_list[2]
activation = activations[0]

results = []

def build_model(window_size, neurons, activation):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(neurons, activation=activation),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


    #   for neurons in neurons_list:
    #   for activation in activations:
for data_type, data_label in [('p', 'Original'), ('p_normalized', 'Normalized')]:
        # data_type = 'p_normalized'
        # data_label = 'Normalized'

    X_train, y_train = create_sequences(train_data[data_type], window_size)
    X_val, y_val = create_sequences(val_data[data_type], window_size)
    X_test, y_test = create_sequences(test_data[data_type], window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_model(window_size, neurons, activation)

    history = model.fit(X_train, y_train, epochs=300, batch_size=32,
                            validation_data=(X_val, y_val), verbose=0)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    if data_label == 'Normalized':
        mse = mse * (std_temp ** 2)

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

best_model_params = min(results, key=lambda x: round(x[4]))
print("Best model parameters:", best_model_params)



