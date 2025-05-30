from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

df = pd.read_csv("dataset/news_dataset.csv")
dataset = Dataset.from_pandas(df)

# stratify_by_column 옵션 제거: 'label' 열이 ClassLabel 형식이 아니라 Value 형식임
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
valid_ds = dataset["test"]

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
valid_ds = valid_ds.map(tokenize_fn, batched=True)

# 데이터셋에 존재하는 컬럼만 제거합니다
train_ds = train_ds.remove_columns(["text"])
valid_ds = valid_ds.remove_columns(["text"])
train_ds.set_format("torch")
valid_ds.set_format("torch")

NUM_LABELS = len(df["label"].unique())

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# 모델을 지정된 디바이스로 이동
model.to(device)
print(f"모델이 {device}로 이동되었습니다.")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="news-finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # GPU 사용 시 학습 속도 향상을 위한 설정

    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

trainer.evaluate()

trainer.save_model("news-finetuned-model")
tokenizer.save_pretrained("news-finetuned-model")