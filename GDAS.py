import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def bootstrap_variance(accuracies):
    """
    Tính phương sai bootstrap từ danh sách độ chính xác.
    """
    B = len(accuracies)
    mean_acc = np.mean(accuracies)
    variance = sum([(acc - mean_acc) ** 2 for acc in accuracies]) / (B - 1)
    return variance


def GDAS(X, y, model, epsilon=0.01, delta=0.05, bootstrap_iterations=100):
    """
    Thực hiện thuật toán Generalized Dynamic Adaptive Sampling (GDAS)

    Parameters:
        X: Dữ liệu đầu vào
        y: Nhãn
        model: Mô hình học máy (ví dụ: DecisionTreeClassifier)
        epsilon: Ngưỡng sai số cho hội tụ
        delta: Xác suất chấp nhận sai số
        bootstrap_iterations: số lượng bootstrap để ước lượng phương sai

    Returns:
        Danh sách các mẫu, độ chính xác ứng với mỗi bước, tổng số mẫu đã dùng
    """
    n_total = len(X)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    results = []
    used_indices = set()

    # Bắt đầu với 10% dữ liệu
    sample_size = max(int(0.1 * n_total), 10)
    prev_acc = 0.0
    converged = False
    step = 1

    while not converged and sample_size < n_total:
        current_indices = np.random.choice(list(set(indices) - used_indices), sample_size, replace=False)
        used_indices.update(current_indices)
        X_sample, y_sample = X[current_indices], y[current_indices]

        # Huấn luyện và tính độ chính xác
        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_sample)
        current_acc = accuracy_score(y_sample, y_pred)

        # Tính phương sai bootstrap
        bootstrapped_accs = []
        for _ in range(bootstrap_iterations):
            X_bs, y_bs = resample(X_sample, y_sample)
            model.fit(X_bs, y_bs)
            y_bs_pred = model.predict(X_bs)
            acc_bs = accuracy_score(y_bs, y_bs_pred)
            bootstrapped_accs.append(acc_bs)

        var_boot = bootstrap_variance(bootstrapped_accs)
        p_hat = abs(current_acc - prev_acc)

        # Tính số mẫu m cần thêm theo BĐT Chebyshev
        if p_hat != 0:
            m_required = int((var_boot / (delta * (epsilon ** 2))) + 1)
        else:
            m_required = 0

        # Lưu kết quả
        results.append({
            'step': step,
            'sample_size': len(used_indices),
            'accuracy': current_acc,
            'variance': var_boot,
            'p_hat': p_hat
        })

        print(
            f"[Step {step}] Sample Size: {len(used_indices)} | Accuracy: {current_acc:.4f} | p̂: {p_hat:.4f} | σ²_boot: {var_boot:.4f}")

        # Kiểm tra hội tụ
        if (p_hat < epsilon) and (var_boot / (bootstrap_iterations * (epsilon ** 2)) <= delta):
            converged = True
        else:
            sample_size = m_required
            prev_acc = current_acc
            step += 1

    return results


def GDAS_with_samples(X, y, model, epsilon=0.01, delta=0.05, bootstrap_iterations=100):
    n_total = len(X)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    results = []
    used_indices = set()
    sample_snapshots = []

    sample_size = max(int(0.1 * n_total), 10)
    prev_acc = 0.0
    converged = False
    step = 1

    while not converged and len(used_indices) + sample_size <= n_total:
        current_indices = np.random.choice(list(set(indices) - used_indices), sample_size, replace=False)
        used_indices.update(current_indices)
        X_sample, y_sample = X[list(used_indices)], y[list(used_indices)]

        model.fit(X_sample, y_sample)
        y_pred = model.predict(X_sample)
        current_acc = accuracy_score(y_sample, y_pred)

        bootstrapped_accs = []
        for _ in range(bootstrap_iterations):
            X_bs, y_bs = resample(X_sample, y_sample)
            model.fit(X_bs, y_bs)
            y_bs_pred = model.predict(X_bs)
            acc_bs = accuracy_score(y_bs, y_bs_pred)
            bootstrapped_accs.append(acc_bs)

        var_boot = bootstrap_variance(bootstrapped_accs)
        p_hat = abs(current_acc - prev_acc)

        if p_hat != 0:
            m_required = int((var_boot / (delta * (epsilon ** 2))) + 1)
        else:
            m_required = 0

        # Lưu thông tin
        results.append({
            'step': step,
            'sample_size': len(used_indices),
            'accuracy': current_acc,
            'variance': var_boot,
            'p_hat': p_hat
        })
        sample_snapshots.append(sorted(list(used_indices)))  # lưu các chỉ số mẫu

        print(
            f"[Step {step}] Sample Size: {len(used_indices)} | Accuracy: {current_acc:.4f} | p̂: {p_hat:.4f} | σ²_boot: {var_boot:.4f}")

        if (p_hat < epsilon) and (var_boot / (bootstrap_iterations * (epsilon ** 2)) <= delta):
            converged = True
        else:
            sample_size = m_required
            prev_acc = current_acc
            step += 1

    return results, sample_snapshots

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Tải dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

model = DecisionTreeClassifier(random_state=42)

# Gọi hàm GDAS có lưu mẫu
results, sample_snapshots = GDAS_with_samples(X, y, model)
#
# # In ra các chỉ số mẫu từng bước
# for i, sample in enumerate(sample_snapshots, 1):
#     print(f"\n[Sample Step {i}] - Sample Size: {len(sample)}")
#     print("Indices:", sample)
#
# # Dữ liệu được chọn ở bước cuối
# final_sample_indices = sample_snapshots[-1]
# X_cut = X[final_sample_indices]
# y_cut = y[final_sample_indices]
#
# print("X_cut shape:", X_cut.shape)   # ví dụ: (48, 4)
# print("y_cut distribution:", np.bincount(y_cut))

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Tải dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

# Phân bố lớp trong toàn bộ dữ liệu
original_counts = np.bincount(y)

# Phân bố lớp sau khi cắt bằng GDAS
final_sample_indices = sample_snapshots[-1]
y_cut = y[final_sample_indices]
cut_counts = np.bincount(y_cut)

# In phân bố lớp
print("🌼 Phân bố lớp trong toàn bộ dữ liệu:")
for i, c in enumerate(original_counts):
    print(f" - {class_names[i]}: {c} mẫu")

print("\n✂️ Phân bố lớp sau khi cắt bằng GDAS:")
for i, c in enumerate(cut_counts):
    print(f" - {class_names[i]}: {c} mẫu")

# x_labels = class_names
# x = np.arange(len(x_labels))  # [0, 1, 2]
#
# plt.bar(x - 0.15, original_counts, width=0.3, label='Dữ liệu gốc')
# plt.bar(x + 0.15, cut_counts, width=0.3, label='Dữ liệu cắt')
#
# plt.xticks(x, x_labels)
# plt.ylabel("Số lượng mẫu")
# plt.title("So sánh phân bố lớp trước và sau khi cắt bằng GDAS")
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.show()


import pandas as pd

# Tạo DataFrame từ dữ liệu đã cắt
X_cut = X[final_sample_indices]
y_cut = y[final_sample_indices]
df_cut = pd.DataFrame(X_cut, columns=iris.feature_names)
df_cut['target'] = y_cut
df_cut['target_name'] = df_cut['target'].apply(lambda x: class_names[x])

# Lưu ra file CSV
df_cut.to_csv("iris_gdas_sample.csv", index=False, encoding='utf-8-sig')
print("Đã lưu tập dữ liệu đã cắt vào 'iris_gdas_sample.csv'")



