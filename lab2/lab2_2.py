import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tabulate import tabulate
from scipy.interpolate import griddata

np.random.seed(1)

n_samples = 200
t = np.random.uniform(0, 10, n_samples)
P = np.random.uniform(500, 3000, n_samples)

noise = np.random.normal(0, 1, n_samples)  # np.random.uniform(-1, 1, n_samples)
T = t * P + noise

scaler = StandardScaler()
X = np.column_stack((t, P))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, T, test_size=0.3, random_state=1)

model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=1000, verbose=0)

y_pred = model.predict(X_test)

X_test_unscaled = scaler.inverse_transform(X_test)
t_test = X_test_unscaled[:, 0]
P_test = X_test_unscaled[:, 1]
y_test_pred = y_pred.flatten()

headers = ["t", "P", "Очікуване", "Передбачене"]
data = list(zip(X_test_unscaled[:, 0], X_test_unscaled[:, 1], y_test, y_pred))
print(tabulate(data, headers=headers, tablefmt='grid', floatfmt=".4f"))

r2 = r2_score(y_test, y_pred)
print(f'Коефіцієнт детермінації R^2: {r2:.4f}')

fig = plt.figure(figsize=(14, 6))

# Графік 1 - Точковий графік
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(t_test, P_test, y_test, label="Очікуване", color='red', alpha=0.6)
ax1.scatter(t_test, P_test, y_test_pred, label="Передбачене", color='blue', alpha=0.6)
ax1.set_xlabel("t")
ax1.set_ylabel("P")
ax1.set_zlabel("T")
ax1.set_title("3D Точковий графік")
ax1.legend()

# Графік 2 - Площини
ax2 = fig.add_subplot(122, projection='3d')
grid_t, grid_P = np.meshgrid(np.linspace(min(t_test), max(t_test), 30),
                             np.linspace(min(P_test), max(P_test), 30))
grid_T_actual = griddata((t_test, P_test), y_test, (grid_t, grid_P), method='cubic')
grid_T_pred = griddata((t_test, P_test), y_test_pred, (grid_t, grid_P), method='cubic')

ax2.plot_surface(grid_t, grid_P, grid_T_actual, cmap='Reds', alpha=0.6, edgecolor='none', label='Очікуване')
ax2.plot_surface(grid_t, grid_P, grid_T_pred, cmap='Blues', alpha=0.6, edgecolor='none', label='Передбачене')
ax2.set_xlabel("t")
ax2.set_ylabel("P")
ax2.set_zlabel("T")
ax2.set_title("Площини фактичних і передбачених значень")

plt.show()