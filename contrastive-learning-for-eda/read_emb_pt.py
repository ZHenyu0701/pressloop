import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 读取 embedding 文件
emb_path = "/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1/embeddings.pt"
label_path = "/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1/labels.pt"

emb = torch.load(emb_path)
labels = torch.load(label_path).numpy()

print(type(emb))
print(emb.shape)
print(labels.shape)

# 👉 转 numpy
emb_np = emb.numpy()

# =========================
# 🔥 加在这里（分类评估）
# =========================
X = emb_np
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
# =========================

# PCA 可视化
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(emb_np)

plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, s=2, cmap="viridis")
plt.colorbar()
plt.title("PCA of embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()