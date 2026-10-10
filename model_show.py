import torch
from collections import OrderedDict

PT_PATH = "outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_15_2026_102142_3/best_model.pt"

obj = torch.load(PT_PATH, map_location="cpu")

print("Loaded:", PT_PATH)
print("Top-level type:", type(obj))

# 情况 A：直接保存了完整模型（少见，但可能）
if isinstance(obj, torch.nn.Module):
    print("\n✅ This .pt contains a full nn.Module. Model structure:\n")
    print(obj)
    raise SystemExit

# 情况 B：checkpoint dict（常见：包含 state_dict/epoch/...）
state_dict = None
if isinstance(obj, dict):
    print("\nCheckpoint keys:", list(obj.keys())[:30], ("..." if len(obj.keys()) > 30 else ""))
    if "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    else:
        # 有些项目直接把参数字典放在最外层
        # 这里用“值是否是 Tensor”来猜
        if all(torch.is_tensor(v) for v in obj.values()):
            state_dict = obj

# 情况 C：OrderedDict（最常见：纯 state_dict）
if isinstance(obj, OrderedDict):
    state_dict = obj

if state_dict is None:
    print("\n❌ Could not find a state_dict-like object in this .pt.")
    print("Try printing repr(obj) or check your saving code.")
    raise SystemExit

print("\n✅ Detected state_dict.")
print("Num tensors:", len(state_dict))

# 只打印 key + shape（不会再刷屏）
print("\n--- First 40 params (key -> shape) ---")
for i, (k, v) in enumerate(state_dict.items()):
    print(f"{i:03d} {k:60s} {tuple(v.shape)}")
    if i >= 39:
        break

# 额外：统计前缀，帮你确认 encoder/head 这些模块
from collections import Counter
prefixes = [k.split(".")[0] for k in state_dict.keys()]
cnt = Counter(prefixes)
print("\n--- Top-level key prefixes (counts) ---")
for p, c in cnt.most_common():
    print(f"{p:25s} {c}")