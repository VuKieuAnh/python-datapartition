import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

# 3. Thêm cột ID (dùng để hash)
df['id'] = range(len(df))

# 4. Chia tập train/test
df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)

# 5. Hash-based partitioning: hash ID % k
k = 3
df_train['hash_bin'] = df_train['id'].apply(lambda x: hash(x) % k)

# 6. Tạo partitions
partitions = [group.drop(columns=['hash_bin']) for _, group in df_train.groupby('hash_bin')]

# 7. Huấn luyện và đánh giá
models = []
accuracies = []

for i, part in enumerate(partitions):
    X_part = part.drop(columns=['target', 'id'])
    y_part = part['target']

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_part, y_part)

    y_pred = clf.predict(df_test.drop(columns=['target', 'id']))
    acc = accuracy_score(df_test['target'], y_pred)

    models.append(clf)
    accuracies.append(acc)

    print(f"Partition {i+1}")
    print(f"Số dòng: {len(part)}")
    print(f"Phân phối nhãn: {dict(Counter(y_part))}")
    print(f"Accuracy trên test: {acc:.4f}\n")

# 8. Tổng hợp kết quả
mean_acc = np.mean(accuracies)
print("Trung bình độ chính xác qua các partition:", round(mean_acc, 4))
