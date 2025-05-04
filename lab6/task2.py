import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import evaluate

DATA_DIR = 'D:/Python/NeuralN/lab6'
train_file = os.path.join(DATA_DIR, 'train.txt')
test_file = os.path.join(DATA_DIR, 'test.txt')
val_file = os.path.join(DATA_DIR, 'val.txt')

def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['text', 'label'])
    return df

train_df = load_data(train_file)
val_df = load_data(val_file)
test_df = load_data(test_file)

def preprocess_text(texts, method='none'):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    processed_texts = []
    for text in texts:
        tokens = word_tokenize(str(text).lower())
        if method == 'lemmatize':
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        elif method == 'stem':
            tokens = [stemmer.stem(token) for token in tokens]
        processed_texts.append(' '.join(tokens))
    return processed_texts

def choose_preprocess_method():
    print("Виберіть метод попередньої обробки тексту:")
    print("1 - Нічого не робити")
    print("2 - Лематизація")
    print("3 - Стемінг")
    choice = input("Ваш вибір (1/2/3): ").strip()
    return {'1': 'none', '2': 'lemmatize', '3': 'stem'}.get(choice, 'none')

PREPROCESS_METHOD = choose_preprocess_method()
train_df['text'] = preprocess_text(train_df['text'], PREPROCESS_METHOD)
val_df['text'] = preprocess_text(val_df['text'], PREPROCESS_METHOD)
test_df['text'] = preprocess_text(test_df['text'], PREPROCESS_METHOD)

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['label'])
val_df['label'] = label_encoder.transform(val_df['label'])
test_df['label'] = label_encoder.transform(test_df['label'])

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "youscan/ukr-roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

training_args = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            annot_kws={"size": 12})
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.title('Confusion Matrix', fontsize=16, pad=20)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

plt.tight_layout()
plt.show()

def interactive_test(model, tokenizer, label_encoder, preprocess_method='none'):
    print("\nІнтерактивне тестування (введіть 'quit' для виходу):")
    while True:
        text = input("\nВведіть текст: ")
        if text.lower() == 'quit':
            break
        processed = preprocess_text([text], method=preprocess_method)
        inputs = tokenizer(processed, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=1).detach().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        print(f"\nКлас: {pred_label}")
        print("Ймовірності:")
        for i, prob in enumerate(probs):
            print(f"  {label_encoder.classes_[i]}: {prob:.4f}")


train_history = trainer.state.log_history

epochs = []
train_loss = []
eval_loss = []
eval_accuracy = []
eval_f1 = []

for log in train_history:
    if "epoch" in log:
        if "loss" in log and "eval_loss" not in log:
            train_loss.append(log["loss"])
        if "eval_loss" in log:
            eval_loss.append(log["eval_loss"])
        if "eval_accuracy" in log:
            eval_accuracy.append(log["eval_accuracy"])
        if "eval_f1" in log:
            eval_f1.append(log["eval_f1"])
        epochs.append(log["epoch"])

history = {
    'loss': train_loss,
    'val_loss': eval_loss,
    'accuracy': eval_accuracy,
    'val_accuracy': eval_accuracy
}


def plot_training_history(history):
    plt.figure(figsize=(6, 4))

    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

interactive_test(model, tokenizer, label_encoder, preprocess_method=PREPROCESS_METHOD)