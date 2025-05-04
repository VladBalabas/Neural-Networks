import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Embedding, LSTM, Dropout, Dense
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

def choose_preprocess_method():
    print("Виберіть метод попередньої обробки тексту:")
    print("1 - Нічого не робити")
    print("2 - Лематизація")
    print("3 - Стемінг")

    choice = input("Ваш вибір (1/2/3): ").strip()
    if choice == '2':
        return 'lemmatize'
    elif choice == '3':
        return 'stem'
    else:
        return 'none'


PREPROCESS_METHOD = choose_preprocess_method()

MAX_LEN = 8
DATA_DIR = 'D:/Python/NeuralN/lab6'
EMBEDDING_DIM = 32
RNN_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 20


def load_data_from_files(train_file, test_file, val_file):
    train_data = pd.read_csv(train_file, header=None, names=['text', 'label'])
    test_data = pd.read_csv(test_file, header=None, names=['text', 'label'])
    val_data = pd.read_csv(val_file, header=None, names=['text', 'label'])
    return train_data, test_data, val_data


def preprocess_text(texts, method='none'):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    processed_texts = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        if method == 'lemmatize':
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        elif method == 'stem':
            tokens = [stemmer.stem(token) for token in tokens]
        processed_texts.append(' '.join(tokens))
    return processed_texts


def prepare_data(data, tokenizer=None, label_encoder=None, fit=False, preprocess_method='none'):
    texts = data['text'].astype(str).tolist()
    labels = data['label'].astype(str).tolist()

    texts = preprocess_text(texts, method=preprocess_method)

    if fit or tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    if fit or label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)

    encoded_labels = label_encoder.transform(labels)

    return padded_sequences, encoded_labels, tokenizer, label_encoder


train_file = os.path.join(DATA_DIR, 'train.txt')
test_file = os.path.join(DATA_DIR, 'test.txt')
val_file = os.path.join(DATA_DIR, 'val.txt')

train_data, test_data, val_data = load_data_from_files(train_file, test_file, val_file)

X_train, y_train, tokenizer, label_encoder = prepare_data(train_data, fit=True, preprocess_method=PREPROCESS_METHOD)
X_test, y_test, _, _ = prepare_data(test_data, tokenizer=tokenizer, label_encoder=label_encoder, preprocess_method=PREPROCESS_METHOD)
X_val, y_val, _, _ = prepare_data(val_data, tokenizer=tokenizer, label_encoder=label_encoder, preprocess_method=PREPROCESS_METHOD)

vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_encoder.classes_)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM),
    LSTM(RNN_UNITS, return_sequences=False),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(history)


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    class_names = label_encoder.classes_

    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(12, 10))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 12})

    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)

    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)

    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
    print(f"\nWeighted Average Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


evaluate_model(model, X_test, y_test, label_encoder)


def interactive_test(model, tokenizer, label_encoder, preprocess_method='none'):
    print("\nInteractive testing mode. Enter 'quit' to exit.")
    while True:
        text = input("\nEnter your text: ")
        if text.lower() == 'quit':
            break

        processed_text = preprocess_text([text], method=preprocess_method)
        sequence = tokenizer.texts_to_sequences(processed_text)
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model.predict(padded_sequence)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = label_encoder.inverse_transform(predicted_class)[0]

        print(f"\nInput text: {text}")
        print(f"Predicted class: {predicted_label}")
        print("Class probabilities:")
        for class_name, prob in zip(label_encoder.classes_, prediction[0]):
            print(f"{class_name}: {prob:.4f}")


# interactive_test(model, tokenizer, label_encoder, preprocess_method=PREPROCESS_METHOD)
