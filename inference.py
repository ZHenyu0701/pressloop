import json
import os
import time
from collections import deque

import numpy as np
import torch
from joblib import load

from run_exp import ExpRunner


CKPT_DIR = "/home/pi/stress_model"   # 树莓派文件路径
CKPT_NAME = "best_model.pt"
CLASSIFIER_NAME = "final_model.joblib"
SCALER_NAME = "scaler.joblib"

DEVICE = "cpu"           
WINDOW_SIZE = 40         # 训练时输入长度 (40 / 64 / 100)
SAMPLE_INTERVAL = 0.25   # 采样间隔s
PRED_THRESHOLD = 0.5     # 二分类阈值
USE_HEART_RATE = True


def load_state_dict_clean(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def load_encoder(ckpt_dir, ckpt_name="best_model.pt", device="cpu"):
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"找不到 config.json: {cfg_path}")

    with open(cfg_path, "r") as f:
        config = json.load(f)

    exp = ExpRunner(config, run_num=0)
    model = exp.init_model()

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型权重: {ckpt_path}")

    model = load_state_dict_clean(model, ckpt_path, device)
    model = model.to(device)
    model.eval()
    return model


def load_classifier_and_scaler(ckpt_dir, classifier_name, scaler_name):
    clf_path = os.path.join(ckpt_dir, classifier_name)
    scaler_path = os.path.join(ckpt_dir, scaler_name)

    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"找不到分类器: {clf_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 scaler: {scaler_path}")

    clf = load(clf_path)
    scaler = load(scaler_path)
    return clf, scaler


# ==============================
# 传感器读数，改动
# ==============================
def read_eda():
    """
    eda读取代码，返回float
    """

    # 假数据
    #eda = np.random.random()
    #return float(eda)


def read_heart_rate():
    """
    心率读取代码，返回 float（例如 BPM）
    如果暂时没有心率传感器，可以先返回 None，分类器会退化为只用 EDA embedding。
    """

    # 假数据
    #hr = 72.0
    #return float(hr)
    return None




# 预处理

def preprocess_window(window):
    """
    窗口数据处理模型输入
    补上额外预处理
    """
    x = np.array(window, dtype=np.float32)

    return x



# 提取 embedding
@torch.no_grad()
def extract_embedding(model, x_np, device="cpu"):
    """
    x_np: shape [WINDOW_SIZE]
    返回 shape [1, D] 的 embedding
    """
    x = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)

    if hasattr(model, "encoder_net"):
        emb = model.encoder_net(x)
    else:
        raise RuntimeError("当前模型没有 encoder_net，无法提取 embedding")

    return emb.detach().cpu().numpy()


def build_classifier_features(model, x_np, heart_rate=None, device="cpu"):
    """
    返回给 sklearn 分类器的特征。
    默认是 encoder embedding；如果提供 heart_rate，则拼接成 [embedding, hr]。
    """
    emb = extract_embedding(model, x_np, device=device)

    if heart_rate is None:
        return emb

    hr_feat = np.array([[float(heart_rate)]], dtype=np.float32)
    return np.concatenate([emb, hr_feat], axis=1)


# 6. 分类
def predict_stress(model, clf, scaler, x_np, heart_rate=None, device="cpu", threshold=0.5):
    features = build_classifier_features(
        model=model,
        x_np=x_np,
        heart_rate=heart_rate,
        device=device,
    )
    emb_scaled = scaler.transform(features)

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(emb_scaled)[0]
        # 默认二分类时取类别1的概率
        stress_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
        pred = int(stress_prob >= threshold)
        return pred, stress_prob
    else:
        pred = int(clf.predict(emb_scaled)[0])
        return pred, float(pred)




def main():
    print("正在加载 encoder...")
    model = load_encoder(CKPT_DIR, CKPT_NAME, DEVICE)

    print("正在加载分类器和 scaler...")
    clf, scaler = load_classifier_and_scaler(CKPT_DIR, CLASSIFIER_NAME, SCALER_NAME)

    print("加载完成，开始实时推理。")
    print(f"WINDOW_SIZE={WINDOW_SIZE}, SAMPLE_INTERVAL={SAMPLE_INTERVAL}s")

    eda_buffer = deque(maxlen=WINDOW_SIZE)

    while True:
        try:
            eda = read_eda()
            heart_rate = read_heart_rate() if USE_HEART_RATE else None
            eda_buffer.append(eda)

            # 打印当前采样值
            print(f"EDA: {eda:.4f}")
            if heart_rate is not None:
                print(f"HR: {heart_rate:.2f}")

            if len(eda_buffer) == WINDOW_SIZE:
                x_np = preprocess_window(list(eda_buffer))
                pred, stress_prob = predict_stress(
                    model=model,
                    clf=clf,
                    scaler=scaler,
                    x_np=x_np,
                    heart_rate=heart_rate,
                    device=DEVICE,
                    threshold=PRED_THRESHOLD
                )

                if pred == 1:
                    print(f"结果: 有压力 | 概率={stress_prob:.3f}")
                else:
                    print(f"结果: 无压力 | 概率={stress_prob:.3f}")

            time.sleep(SAMPLE_INTERVAL)

        except KeyboardInterrupt:
            print("\n已停止。")
            break
        except Exception as e:
            print(f"运行报错: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
