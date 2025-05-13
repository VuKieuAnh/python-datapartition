import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
freature_name = iris.feature_names

df = pd.DataFrame(X, columns=freature_name)
df['target']=y

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

k = 8
partitions = np.array_split(df_train.sample(frac=1, random_state=42), k)

# # 5. Hiển thị nội dung từng phần sau khi chia
# for i, part in enumerate(partitions):
#     print(f"\n📦 Partition {i+1} - Số dòng: {len(part)}")
#     print(part.head())  # In 5 dòng đầu tiên của mỗi partition

# 5. Huấn luyện mô hình Decision Tree trên từng phần
models = []
accuracies = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


for i, part in enumerate(partitions):
    X_part = part.drop(columns='target')
    y_part = part['target']

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_part, y_part)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    models.append(clf)
    accuracies.append(acc)

    print(f"Partition {i + 1} - Accuracy: {acc:.4f}")

# 6. Tổng hợp kết quả
mean_acc = np.mean(accuracies)
print("Trung bình độ chính xác qua các partition:", round(mean_acc, 4))
