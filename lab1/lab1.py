import random
import numpy as np
import matplotlib.pyplot as plt
import csv
from tabulate import tabulate


class NeuralNetwork:
    def __init__(self, learn_file, control_file):
        self.theta = 0  # 850000
        self.learning_rate = 0.1
        self.learn_data, self.learn_answers = self.read_csv_data(learn_file)
        self.control_data, self.control_answers = self.read_csv_data(control_file)

        # perfect: [-1, 400, 2720021]
        self.weights = [0, 0, 0.5]

    def normalize_data(self, data, learning=True):
        columns = list(zip(*data))

        min_vals = [min(col) for col in columns[:-1]]
        max_vals = [max(col) for col in columns[:-1]]

        normalized_data = [
            [(x - min_vals[j]) / (max_vals[j] - min_vals[j]) if max_vals[j] != min_vals[j] else 0
             for j, x in enumerate(row[:-1])] + [row[-1]]
            for row in data
        ]
        if learning:
            self.learn_data = normalized_data
        else:
            self.control_data = normalized_data

    def activation_function(self, x):
        return 1 if x < self.theta else 2

    def calculate_sum(self, x_vector):
        return sum(w * x for w, x in zip(self.weights, x_vector))

    def train(self, randomize=False):
        for i in range(10 * len(self.learn_data) if randomize else len(self.learn_data)):
            print(self.weights)
            index = random.randint(0, len(self.learn_data)-1) if randomize else i

            output = self.activation_function(self.calculate_sum(self.learn_data[index]))

            if output != self.learn_answers[index]:
                for j in range(len(self.weights)):
                    self.weights[j] += self.learning_rate * (self.learn_answers[index] - output) * self.learn_data[index][j]

        print(f"Weights after training: {self.weights}")

    def test(self, test_data, test_answers):
        table = []
        correct_predictions = 0

        for i in range(len(test_data)):
            x_vector = test_data[i]
            expected_class = test_answers[i]
            sum_value = self.calculate_sum(x_vector)
            predicted_class = self.activation_function(sum_value)

            if predicted_class == expected_class:
                correct_predictions += 1

            row = [
                i + 1, x_vector[0], x_vector[1], x_vector[2], sum_value, predicted_class, expected_class
            ]
            table.append(row)

        headers = ["№", "Arg 1", "Arg 2", "Arg 3", "Sum", "Predicted Class", "Actual Class"]
        print(tabulate(table, headers=headers, tablefmt="grid"))

        accuracy = correct_predictions / len(test_data) * 100
        print(f"\nТочність: {accuracy:.2f}%\n")

    def read_csv_data(self, filename):
        data = []
        labels = []

        with open(filename, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append([int(row[0]), int(row[1]), -1])
                labels.append(int(row[2]))
        return data, labels

    def visualize_data(self, data, labels, weights, title="Visualization"):
        data = np.array(data)
        labels = np.array(labels)

        class_1 = data[labels == 1]
        class_2 = data[labels == 2]

        plt.figure(figsize=(8, 6))
        plt.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Клас 1')
        plt.scatter(class_2[:, 0], class_2[:, 1], color='red', label='Клас 2')

        x_vals = np.linspace(min(data[:, 0]), max(data[:, 0]), 100)
        y_vals = (-x_vals * weights[0] + weights[2] + self.theta) / weights[1]

        plt.plot(x_vals, y_vals, color='green')

        plt.xlabel("Ознака 1")
        plt.ylabel("Ознака 2")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    while True:
        print("Оберіть файл для навчання:")
        print("1 - lab1_learn.csv")
        print("2 - lab1_learn_add.csv")
        choice = input("Введіть номер файлу: ").strip()

        if choice == "1":
            learn_file = "lab1_learn.csv"
        elif choice == "2":
            learn_file = "lab1_learn_add.csv"
        else:
            print("Некоректне введення, використовується файл за замовчуванням: lab1_learn.csv")
            learn_file = "lab1_learn.csv"

        control_file = "lab1_control.csv"

        randomize = input("Використовувати випадковий порядок подачі даних? (1 - Так / 2 - Ні): ").strip() == "1"
        normalize = input("Нормалізувати дані? (1 - Так / 2 - Ні): ").strip() == "1"

        nn = NeuralNetwork(learn_file, control_file)

        if normalize:
            nn.normalize_data(nn.learn_data)
            nn.normalize_data(nn.control_data, learning=False)

        nn.train(randomize=randomize)

        nn.visualize_data(nn.learn_data, nn.learn_answers, nn.weights, title="Навчальна вибірка")
        nn.test(test_data=nn.learn_data, test_answers=nn.learn_answers)

        nn.visualize_data(nn.control_data, nn.control_answers, nn.weights, title="Контрольна вибірка")
        nn.test(test_data=nn.control_data, test_answers=nn.control_answers)

        repeat = input("\nЗапустити знову? (1 - Так / 2 - Вийти): ").strip()
        if repeat != "1":
            break


if __name__ == "__main__":
    main()
