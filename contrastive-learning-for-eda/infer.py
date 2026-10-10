import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from run_exp import ExpRunner


def move_to_device(x, device):
    if torch.is_tensor(x):
        x = x.to(device, non_blocking=True)
        if x.is_floating_point():
            x = x.float()
        return x
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    return x


def load_state_dict_clean(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


@torch.no_grad()
def extract_embeddings(model, dataset, batch_size, num_workers, device):
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=device.startswith("cuda"),
    )

    all_emb = []

    for batch in loader:
        batch = move_to_device(batch, device)

        # 尝试提取输入 x
        x = batch
        if isinstance(batch, dict):
            for key in ["x", "inputs", "data", "signal"]:
                if key in batch:
                    x = batch[key]
                    break
        elif isinstance(batch, (list, tuple)):
            x = batch[0]

        # 优先用 encoder_net（对比学习标准做法）
        if hasattr(model, "encoder_net"):
            emb = model.encoder_net(x)
        else:
            # 兜底：直接 forward
            out = model(x)
            if isinstance(out, (tuple, list)):
                emb = next((o for o in out if torch.is_tensor(o) and o.dim() > 1), out[-1])
            else:
                emb = out

        all_emb.append(emb.detach().cpu())

    return torch.cat(all_emb, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--ckpt_name", type=str, default="best_model.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="embeddings.pt")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # 读取 config
    cfg_path = os.path.join(args.ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到 config.json: {cfg_path}")

    with open(cfg_path, "r") as f:
        config = json.load(f)

    # 自动选择可用 split
    if args.split == "test" and "test_dataset_name" not in config:
        print("[warn] test 不存在，改用 val")
        args.split = "val"
    if args.split == "val" and "val_dataset_name" not in config:
        print("[warn] val 不存在，改用 train")
        args.split = "train"

    config["do_train"] = args.split == "train"
    config["do_val"] = args.split == "val"
    config["do_test"] = args.split == "test"

    # 构建实验对象
    exp = ExpRunner(config, run_num=0)
    model = exp.init_model()

    if args.split == "train":
        dataset = exp.train_dataset
    elif args.split == "val":
        dataset = exp.val_dataset
    else:
        dataset = exp.test_dataset

    if dataset is None:
        raise RuntimeError(f"{args.split}_dataset 为 None")

    # 加载权重
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    model = load_state_dict_clean(model, ckpt_path, device)

    # 提取 embedding
    embeddings = extract_embeddings(
        model,
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    out_path = os.path.join(args.ckpt_dir, args.out)
    torch.save(embeddings, out_path)

    print(f"[OK] embeddings shape = {embeddings.shape}")
    print(f"[OK] saved to {out_path}")


if __name__ == "__main__":
    main()