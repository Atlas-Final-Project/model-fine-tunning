from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 모델과 토크나이저 로드
model_path = "news-finetuned-model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 모델을 평가 모드로 설정하고 디바이스로 이동
model.eval()
model.to(device)

# 테스트 데이터셋 로드 (여기서는 검증 데이터셋을 사용)
df = pd.read_csv("dataset/news_dataset.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
test_ds = dataset["test"]

# 토큰화 함수
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 데이터셋 토큰화
test_ds = test_ds.map(tokenize_fn, batched=True)

# PyTorch 텐서 형식으로 설정 (필요한 컬럼만 유지)
columns_to_keep = ['input_ids', 'attention_mask', 'label']
if 'token_type_ids' in test_ds.column_names:
    columns_to_keep.append('token_type_ids')

# 데이터셋에서 text 컬럼 제거 (이미 토큰화 되었으므로)
if 'text' in test_ds.column_names:
    test_ds = test_ds.remove_columns(['text'])

# PyTorch 텐서 형식으로 설정
test_ds.set_format(type='torch', columns=columns_to_keep)

# 예측 수행
all_preds = []
all_labels = []

print("테스트 데이터셋에 대한 예측 수행 중...")
for i in tqdm(range(len(test_ds))):
    # 샘플 가져오기
    sample = test_ds[i]
    
    # 입력 데이터 준비 - 이미 torch 텐서 형식임
    inputs = {}
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        if k in sample:
            # 배치 차원 추가 및 장치로 이동
            inputs[k] = sample[k].unsqueeze(0).to(device)
    
    # 레이블 저장
    label = sample['label'].item()  # PyTorch 텐서에서 Python 정수로 변환
    all_labels.append(label)
    
    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    
    all_preds.append(pred)

# 모델 성능 평가
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
report = classification_report(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f"\n정확도: {accuracy:.4f}")
print(f"F1 점수: {f1:.4f}")
print("\n분류 보고서:")
print(report)

# 레이블명 가져오기 (레이블이 숫자로 표현된 경우)
label_names = sorted(df['label'].unique())

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title('혼동 행렬')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n혼동 행렬이 'confusion_matrix.png' 파일로 저장되었습니다.")

# 실제 텍스트 예시로 예측 테스트
test_examples = [
    "Massive earthquake shakes California coast",
    "New vaccine shows promising results in clinical trials",
    "Stock market reaches all-time high",
    "Scientists discover new species in Amazon rainforest",
    "Government announces new tax policy"
]

print("\n예시 텍스트에 대한 예측:")
for text in test_examples:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    
    # 레이블 이름 출력 (사용 가능한 경우)
    label_name = label_names[pred] if len(label_names) > pred else f"LABEL_{pred}"
    print(f"텍스트: {text}")
    print(f"예측 레이블: {label_name}\n")
