import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. 读取 embedding 文件
path = "/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1/embeddings.pt"
emb = torch.load(path)

# 加载真实标签
label_path = "/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1/labels.pt"
labels = torch.load(label_path).numpy()

# 2. 打印基本信息
print(type(emb))
print(emb.shape)

# 3. 转成 numpy
emb_np = emb.numpy()

# 4. 只取一部分做 t-SNE
emb_small = emb_np[:5000]
labels_small = labels[:5000]

# 5. t-SNE降到2维
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(emb_small)

# 6. 画图
plt.figure(figsize=(8, 6))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels_small, s=2, cmap="viridis")
plt.colorbar()
plt.title("t-SNE of embeddings")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.show()