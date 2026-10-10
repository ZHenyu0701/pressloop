import json
import os
import pickle
import sys

import neurokit2 as nk
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from run_exp import ExpRunner


START_DT = pd.Timestamp("2017-11-28 00:00:00", tz="UTC")
EDA_SAMPLE_RATE = 4
WESAD_WINDOW_SIZE = 240


def _load_pickle(raw_data_dir, subject_id):
    pkl_path = os.path.join(raw_data_dir, subject_id, f"{subject_id}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing WESAD pickle: {pkl_path}")
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def _get_signal_and_rate(subject_data, signal_source):
    source = signal_source.lower()
    if source == "bvp":
        signal = subject_data["signal"]["wrist"]["BVP"]
        sample_rate = 64
        process_kind = "ppg"
    elif source == "ecg":
        signal = subject_data["signal"]["chest"]["ECG"]
        sample_rate = 700
        process_kind = "ecg"
    else:
        raise ValueError(f"Unsupported signal_source: {signal_source}")

    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    return signal, sample_rate, process_kind


def _compute_hr_signal(signal, sample_rate, process_kind):
    if process_kind == "ppg":
        processed, info = nk.ppg_process(signal, sampling_rate=sample_rate)
        rate_key = "PPG_Rate"
        peak_key = "PPG_Peaks"
    else:
        processed, info = nk.ecg_process(signal, sampling_rate=sample_rate)
        rate_key = "ECG_Rate"
        peak_key = "ECG_R_Peaks"

    if rate_key in processed:
        rate = processed[rate_key].to_numpy(dtype=np.float64)
    else:
        peaks = info[peak_key]
        rate = nk.signal_rate(peaks, sampling_rate=sample_rate, desired_length=len(signal))

    if rate.shape[0] != len(signal):
        rate = np.asarray(rate, dtype=np.float64).reshape(-1)
        if rate.shape[0] != len(signal):
            raise ValueError("Heart-rate signal length does not match source signal length")
    return rate


def _timestamp_to_index(ts, sample_rate):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    delta_seconds = (ts - START_DT).total_seconds()
    return int(round(delta_seconds * sample_rate))


def _window_seconds(window_size=WESAD_WINDOW_SIZE, eda_sample_rate=EDA_SAMPLE_RATE):
    return window_size / eda_sample_rate


def _mean_window_hr(rate_signal, start_ts, sample_rate, window_size=WESAD_WINDOW_SIZE):
    start_idx = _timestamp_to_index(start_ts, sample_rate)
    end_idx = start_idx + int(round(_window_seconds(window_size) * sample_rate))
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(rate_signal))
    if end_idx <= start_idx:
        return np.nan
    hr_window = rate_signal[start_idx:end_idx]
    hr_window = hr_window[np.isfinite(hr_window)]
    if hr_window.size == 0:
        return np.nan
    return float(np.mean(hr_window))


def _load_dataset_from_config(config_path, split):
    with open(config_path, "r") as f:
        config = json.load(f)

    config["do_train"] = split == "train"
    config["do_val"] = split == "val"
    config["do_test"] = split == "test"

    exp = ExpRunner(config, run_num=0)
    if split == "train":
        dataset = exp.train_dataset
    elif split == "val":
        dataset = exp.val_dataset
    else:
        dataset = exp.test_dataset

    if dataset is None:
        raise RuntimeError(f"{split}_dataset is None for config {config_path}")
    return dataset


def build_wesad_hr_features(dataset, raw_data_dir, signal_source="bvp", window_size=WESAD_WINDOW_SIZE):
    subject_ids = sorted({item["subject_id"] for item in dataset.data_list})
    subject_rate_cache = {}
    subject_sample_rate = {}

    for subject_id in subject_ids:
        subject_data = _load_pickle(raw_data_dir, subject_id)
        signal, sample_rate, process_kind = _get_signal_and_rate(subject_data, signal_source)
        rate_signal = _compute_hr_signal(signal, sample_rate, process_kind)
        subject_rate_cache[subject_id] = rate_signal
        subject_sample_rate[subject_id] = sample_rate

    hr_values = []
    for item in dataset.data_list:
        subject_id = item["subject_id"]
        start_ts = item["segment_start_datetime_utc"]
        hr = _mean_window_hr(
            subject_rate_cache[subject_id],
            start_ts,
            subject_sample_rate[subject_id],
            window_size=window_size,
        )
        hr_values.append(hr)

    hr_array = np.asarray(hr_values, dtype=np.float32).reshape(-1, 1)

    if np.isnan(hr_array).any():
        finite_vals = hr_array[np.isfinite(hr_array)]
        if finite_vals.size == 0:
            raise ValueError("All extracted heart-rate values are NaN")
        fill_value = float(np.mean(finite_vals))
        hr_array[~np.isfinite(hr_array)] = fill_value

    return hr_array


def export_wesad_hr_features(config_path, raw_data_dir, split, out_path, signal_source="bvp"):
    dataset = _load_dataset_from_config(config_path, split)
    hr_array = build_wesad_hr_features(dataset, raw_data_dir, signal_source=signal_source)
    torch.save(torch.from_numpy(hr_array), out_path)
    return hr_array
