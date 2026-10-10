import json
import argparse
import ast
import re
from pathlib import Path

import pandas as pd


def _extract_numbers(s: str):
    """Extract all float-like numbers from string."""
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def _is_almost_constant(arr, tol=1e-9):
    if not arr:
        return False
    return (max(arr) - min(arr)) < tol


def _looks_like_loss(arr):
    """
    Heuristic: loss should be positive and not crazy large.
    Adjust bounds if your loss scale is different.
    """
    if not arr:
        return False
    # allow a few outliers
    ok = [x for x in arr if (x is not None and x == x)]
    if not ok:
        return False
    # typical loss: >0, and usually not > 1e3
    frac_reasonable = sum((x > 0 and x < 1e3) for x in ok) / len(ok)
    return frac_reasonable > 0.8


def _cleanup_alternating_constant(nums):
    """
    Fix pattern like:
      [32.0, 4.58, 32.0, 4.47, ...]
    by detecting one parity is constant and dropping it.
    """
    if len(nums) < 6:
        return nums

    odd = nums[0::2]
    even = nums[1::2]

    odd_const = _is_almost_constant(odd)
    even_const = _is_almost_constant(even)

    # Prefer the side that looks like loss and isn't constant
    if odd_const and not even_const and _looks_like_loss(even):
        return even
    if even_const and not odd_const and _looks_like_loss(odd):
        return odd

    # If both look non-constant or both constant, keep original
    return nums


def parse_maybe_serialized(obj):
    """
    Normalize obj into Python types:
    - dict/list kept
    - str that contains JSON / python literal -> parsed
    - otherwise: extract numbers; if alternating constant detected, clean it
    """
    if obj is None:
        return None

    if isinstance(obj, (dict, list, int, float, bool)):
        return obj

    if isinstance(obj, str):
        s = obj.strip()
        if s == "":
            return []

        # Try JSON first
        try:
            parsed = json.loads(s)
            return parsed
        except Exception:
            pass

        # Try python literal
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (dict, list, int, float, bool)):
                return parsed
        except Exception:
            pass

        # Fallback: extract numbers
        nums = _extract_numbers(s)
        if nums:
            nums = _cleanup_alternating_constant(nums)
            return nums

        return obj

    return obj


def to_epoch_loss_rows(loss_data, best_epoch=None):
    loss_data = parse_maybe_serialized(loss_data)
    rows = []

    if loss_data is None:
        return rows

    if isinstance(loss_data, dict):
        for k, v in loss_data.items():
            try:
                epoch = int(k)
            except Exception:
                continue
            rows.append({"epoch": epoch, "train_loss": v, "is_best": epoch == best_epoch})
        rows.sort(key=lambda x: x["epoch"])
        return rows

    if isinstance(loss_data, list):
        if len(loss_data) == 0:
            return rows

        if isinstance(loss_data[0], dict):
            for item in loss_data:
                if not isinstance(item, dict):
                    continue
                epoch = item.get("epoch", item.get("Epoch"))
                loss = item.get("loss", item.get("train_loss", item.get("Loss")))
                if epoch is None:
                    continue
                try:
                    epoch = int(epoch)
                except Exception:
                    continue
                rows.append({"epoch": epoch, "train_loss": loss, "is_best": epoch == best_epoch})
            rows.sort(key=lambda x: x["epoch"])
            return rows

        # list of numbers
        for i, loss in enumerate(loss_data, start=1):
            rows.append({"epoch": i, "train_loss": loss, "is_best": i == best_epoch})
        return rows

    return rows


def to_step_loss_rows(step_data):
    step_data = parse_maybe_serialized(step_data)
    rows = []

    if step_data is None:
        return rows

    if isinstance(step_data, dict):
        for k, v in step_data.items():
            try:
                step = int(k)
            except Exception:
                continue
            rows.append({"step": step, "train_loss": v})
        rows.sort(key=lambda x: x["step"])
        return rows

    if isinstance(step_data, list):
        if len(step_data) == 0:
            return rows

        if isinstance(step_data[0], dict):
            for item in step_data:
                if not isinstance(item, dict):
                    continue
                step = item.get("step", item.get("Step"))
                loss = item.get("loss", item.get("train_loss", item.get("Loss")))
                if step is None:
                    continue
                try:
                    step = int(step)
                except Exception:
                    continue
                rows.append({"step": step, "train_loss": loss})
            rows.sort(key=lambda x: x["step"])
            return rows

        for i, loss in enumerate(step_data, start=1):
            rows.append({"step": i, "train_loss": loss})
        return rows

    return rows


def to_eval_rows(eval_metrics, eval_epochs=None, best_epoch=None):
    eval_metrics = parse_maybe_serialized(eval_metrics)
    eval_epochs = parse_maybe_serialized(eval_epochs)

    rows = []

    if isinstance(eval_metrics, list) and eval_metrics and isinstance(eval_metrics[0], dict):
        for item in eval_metrics:
            row = dict(item)
            if "epoch" not in row and "Epoch" in row:
                row["epoch"] = row.pop("Epoch")
            if "epoch" in row:
                try:
                    row["epoch"] = int(row["epoch"])
                except Exception:
                    pass
            if best_epoch is not None and "epoch" in row:
                row["is_best"] = (row["epoch"] == best_epoch)
            rows.append(row)
        if rows and "epoch" in rows[0]:
            rows.sort(key=lambda x: x.get("epoch", 10**18))
        return rows

    if isinstance(eval_epochs, list) and eval_epochs and isinstance(eval_epochs[0], dict):
        for item in eval_epochs:
            row = dict(item)
            if "epoch" in row:
                try:
                    row["epoch"] = int(row["epoch"])
                except Exception:
                    pass
            if best_epoch is not None and "epoch" in row:
                row["is_best"] = (row["epoch"] == best_epoch)
            rows.append(row)
        if rows and "epoch" in rows[0]:
            rows.sort(key=lambda x: x.get("epoch", 10**18))
        return rows

    if isinstance(eval_metrics, dict):
        for k, v in eval_metrics.items():
            try:
                epoch = int(k)
            except Exception:
                continue
            row = {"epoch": epoch, "is_best": (epoch == best_epoch)}
            if isinstance(v, dict):
                row.update(v)
            else:
                row["metric"] = v
            rows.append(row)
        rows.sort(key=lambda x: x["epoch"])
        return rows

    return rows


def safe_write_csv(df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True, help="path to train_history.json")
    parser.add_argument("--out_dir", type=str, default=".", help="output directory")
    parser.add_argument("--prefix", type=str, default=None, help="prefix for csv filenames")
    parser.add_argument("--debug", action="store_true", help="print raw previews for debugging")
    args = parser.parse_args()

    history_path = Path(args.history)
    with open(history_path, "r") as f:
        data = json.load(f)

    best_epoch = data.get("best_so_far_epoch")

    if args.debug:
        raw = data.get("train_loss_epochs")
        print("DEBUG train_loss_epochs type:", type(raw))
        print("DEBUG train_loss_epochs preview:", repr(raw)[:400])

    out_dir = Path(args.out_dir)
    prefix = args.prefix or history_path.stem

    train_epoch_rows = to_epoch_loss_rows(data.get("train_loss_epochs"), best_epoch=best_epoch)
    df_train_epochs = pd.DataFrame(train_epoch_rows)

    train_step_rows = to_step_loss_rows(data.get("train_loss_steps"))
    df_train_steps = pd.DataFrame(train_step_rows)

    eval_rows = to_eval_rows(
        data.get("eval_metrics"),
        eval_epochs=data.get("eval_epochs"),
        best_epoch=best_epoch,
    )
    df_eval = pd.DataFrame(eval_rows)

    outputs = []

    if not df_train_epochs.empty:
        out_path = safe_write_csv(df_train_epochs, out_dir / f"{prefix}_train_epochs.csv")
        outputs.append(("train_epochs", out_path, df_train_epochs))
    else:
        print("⚠️ No train epoch loss rows parsed (train_loss_epochs).")

    if not df_train_steps.empty:
        out_path = safe_write_csv(df_train_steps, out_dir / f"{prefix}_train_steps.csv")
        outputs.append(("train_steps", out_path, df_train_steps))
    else:
        print("⚠️ No train step loss rows parsed (train_loss_steps).")

    if not df_eval.empty:
        out_path = safe_write_csv(df_eval, out_dir / f"{prefix}_eval.csv")
        outputs.append(("eval", out_path, df_eval))
    else:
        print("⚠️ No eval rows parsed (eval_metrics/eval_epochs).")

    print("\n========== DONE ==========")
    print(f"best_so_far_epoch: {best_epoch}")
    for name, path, df in outputs:
        print(f"✅ {name}: {path}")
        print(df.head())
        print("--------------------------")


if __name__ == "__main__":
    main()