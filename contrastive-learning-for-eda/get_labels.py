import json
import os
import torch
from run_exp import ExpRunner

ckpt_dir = "/Users/daizhenyu/MITmodel/contrastive-learning-for-eda/outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_023723_1"

# 读取 config
cfg_path = os.path.join(ckpt_dir, "config.json")
with open(cfg_path, "r") as f:
    config = json.load(f)

# 因为你之前实际提的是 train
config["do_train"] = True
config["do_val"] = False
config["do_test"] = False

exp = ExpRunner(config, run_num=0)
dataset = exp.train_dataset

labels = []

for i in range(len(dataset)):
    sample = dataset[i]

    if isinstance(sample, dict):
        if "y" in sample:
            y = sample["y"]
        elif "label" in sample:
            y = sample["label"]
        else:
            raise KeyError(f"第 {i} 个 sample 里没有 y 或 label，keys={sample.keys()}")

    elif isinstance(sample, (list, tuple)):
        if len(sample) < 2:
            raise ValueError(f"第 {i} 个 sample 长度不足，无法取 label: {sample}")
        y = sample[1]

    else:
        raise TypeError(f"第 {i} 个 sample 类型不支持: {type(sample)}")

    if torch.is_tensor(y):
        y = y.item()

    labels.append(int(y))

labels = torch.tensor(labels)

print("labels shape:", labels.shape)
print("unique labels:", torch.unique(labels, return_counts=True))

out_path = os.path.join(ckpt_dir, "labels.pt")
torch.save(labels, out_path)
print("saved to:", out_path)