import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# 1. Load dữ liệu
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 2. Tạo DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 3. Chia tập train/test
df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)

# 4. Khởi tạo chia tầng
k = 3
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# 5. Khởi tạo list lưu model và độ chính xác
models = []
accuracies = []

# 6. Lặp qua các partition
for i, (_, idx) in enumerate(skf.split(df_train.drop(columns='target'), df_train['target'])):
    part = df_train.iloc[idx]
    X_part = part.drop(columns='target')
    y_part = part['target']

    # Huấn luyện Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_part, y_part)

    # Dự đoán trên tập test
    y_pred = clf.predict(df_test.drop(columns='target'))
    acc = accuracy_score(df_test['target'], y_pred)

    # Lưu model và độ chính xác
    models.append(clf)
    accuracies.append(acc)

    print(f"Partition {i+1}")
    print(f"Số dòng: {len(part)} - Phân phối nhãn: {dict(Counter(y_part))}")
    print(f"Accuracy trên test: {acc:.4f}\n")

mean_acc = np.mean(accuracies)
print("Trung bình độ chính xác qua các partition:", round(mean_acc, 4))
