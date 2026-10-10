import json

path = "outputs/wesad_contrastive_eval/wesad_contrastive_pretrain/results_Jan_14_2026_234303_0/train_history.json"

with open(path, "r") as f:
    hist = json.load(f)

print(hist.keys())