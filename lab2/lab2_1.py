import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras

np.random.seed(1)
x = np.linspace(-5, 5, 100)
y_true = x**2 - 1 / (np.exp(x + 1) + 1) + np.random.uniform(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y_true, test_size=0.2, random_state=1)

model = keras.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=1000, verbose=0)
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print(f'Коефіцієнт детермінації R^2: {r2:.4f}')

# y_pred_flat = y_pred.flatten()
# y_test_mean = np.mean(y_test)
# ss_residual = np.sum((y_test - y_pred_flat) ** 2)
# ss_total = np.sum((y_test - y_test_mean) ** 2)
# r2_manual = 1 - (ss_residual / ss_total)
# print(f'R^2 (вручну): {r2_manual:.4f}')

plt.scatter(x_train, y_train, label='Тренувальні дані', alpha=0.6)
plt.scatter(x_test, y_test, label='Тестові дані', alpha=0.6, color='red')
plt.plot(x, x**2 - 1 / (np.exp(x + 1) + 1), label='Цільова функція', color='black', linestyle='dashed')
plt.scatter(x_test, y_pred, label='Прогноз моделі', color='green')
plt.legend()
plt.show()
