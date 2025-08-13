# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump

# 1. 간단한 데이터 생성 (실제로는 data/ 폴더에서 불러옵니다)
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})
X = df[['feature1']]
y = df['target']

# 2. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. 학습된 모델을 파일로 저장 (파이프라인의 최종 결과물)
dump(model, 'model.joblib')

print("Model training complete and model.joblib saved.")
