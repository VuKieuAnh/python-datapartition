import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Tải dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# 2. Chia dữ liệu thành train/test (70/30)
df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)

# 3. Tách đặc trưng và nhãn
X_train = df_train.drop(columns='target')
y_train = df_train['target']
X_test = df_test.drop(columns='target')
y_test = df_test['target']

# 4. Huấn luyện mô hình Decision Tree trên toàn bộ dữ liệu huấn luyện
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 5. Dự đoán và đánh giá độ chính xác
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 6. In kết quả
print(f"Độ chính xác khi huấn luyện trên toàn bộ dữ liệu huấn luyện: {accuracy:.4f}")
