import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from tabulate import tabulate

df = pd.read_csv('output.csv')
choosed_column = df['p']

mean_temp = np.mean(choosed_column)
median_temp = np.median(choosed_column)
std_temp = np.std(choosed_column, ddof=1)
min_temp = np.min(choosed_column)
max_temp = np.max(choosed_column)
q1 = np.percentile(choosed_column, 25)
q2 = np.percentile(choosed_column, 50)
q3 = np.percentile(choosed_column, 75)
skewness = stats.skew(choosed_column)
kurtosis = stats.kurtosis(choosed_column)

print("Середнє арифметичне:", round(mean_temp, 2))
print("Медіана:", round(median_temp, 2))
print("Стандартне відхилення:", round(std_temp, 2))
print("Мінімальне значення:", min_temp)
print("Максимальне значення:", max_temp)
print("Квартилі (25%, 50%, 75%):", round(q1, 2), round(q2, 2), round(q3, 2))
print("Коефіцієнт асиметрії:", round(skewness, 2))
print("Куртозис:", round(kurtosis, 2))

plt.figure(figsize=(10, 7))
plt.plot(range(1, len(choosed_column) + 1), choosed_column, linestyle='-', linewidth=2)
plt.title('Динаміка змін тиску')
plt.xlabel('№ Запису')
plt.ylabel('Тиск, Па')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 7))
plt.hist(choosed_column, bins=len(choosed_column), color='black')
plt.title('Гістограма розподілу тиску')
plt.xlabel('Значення тиску')
plt.ylabel('Частота')
plt.grid(axis='y')
plt.show()

normalized_column = (choosed_column - mean_temp) / std_temp
df['p_normalized'] = normalized_column


def create_sequences(data, n):
    sequences = []
    targets = []
    for i in range(len(data) - n):
        sequences.append(data[i:i+n])
        targets.append(data[i+n])
    return np.array(sequences), np.array(targets)


def print_sequences(sequences, targets):
    table_data = [[[float(f"{x:.10f}") for x in seq], f"{target:.10f}"] for seq, target in zip(sequences, targets)]
    headers = ["Послідовність", "Прогнозне значення"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center", numalign="center"))


n = 5
sequences, targets = create_sequences(choosed_column, n)
norm_sequences, norm_targets = create_sequences(df['p_normalized'], n)
print_sequences(sequences, targets)

