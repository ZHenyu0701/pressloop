import argparse
import os

from wesad_hr_features import export_wesad_hr_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--wesad_raw_dir", type=str, required=True)
    parser.add_argument("--signal_source", type=str, default="bvp", choices=["bvp", "ecg"])
    parser.add_argument("--out", type=str, default="heart_rate.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = os.path.join(args.ckpt_dir, "config.json")
    out_path = os.path.join(args.ckpt_dir, args.out)
    hr_array = export_wesad_hr_features(
        config_path=config_path,
        raw_data_dir=args.wesad_raw_dir,
        split=args.split,
        out_path=out_path,
        signal_source=args.signal_source,
    )
    print("heart-rate feature shape:", hr_array.shape)
    print("saved to:", out_path)


if __name__ == "__main__":
    main()
