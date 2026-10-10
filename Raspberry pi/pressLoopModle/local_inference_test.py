import os
import json
import random
import time
from collections import deque

import numpy as np
import torch
from joblib import load

from models.contrastive_model import ContrastiveModel

MODEL_DIR = "."
WINDOW_SIZE = 240
DEVICE = "cpu"

USE_REAL_DATA = True   # False=模拟数据；True=MQTT真实数据
SLEEP_TIME = 0.25       # 模拟测试

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "esp32/eda"

CLASSIFIER_FILE = "RandomForest.joblib"
SCALER_FILE = "scaler.joblib"

latest_eda = None


# MQTT
if USE_REAL_DATA:
    import paho.mqtt.client as mqtt

    def on_connect(client, userdata, flags, rc):
        print("MQTT connected:", rc)
        client.subscribe(MQTT_TOPIC)
        print("Subscribed:", MQTT_TOPIC)

    def on_message(client, userdata, msg):
        global latest_eda
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)

            if msg.topic == MQTT_TOPIC and "eda_uS" in data:
                latest_eda = float(data["eda_uS"])
                print("EDA (real):", latest_eda)

        except Exception as e:
            print("MQTT error:", e)

    def start_mqtt():
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        return client
else:
    def start_mqtt():
        return None

# encoder
def load_encoder():
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)

    model_args = config["model_args"]

    model = ContrastiveModel(
        encoder_name=model_args["encoder_name"],
        encoder_args=model_args["encoder_args"],
        transform_head_name=model_args.get("transform_head_name"),
        transform_head_args=model_args.get("transform_head_args"),
    )

    ckpt = torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        state_dict = ckpt

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


@torch.no_grad()
def extract_embedding(model, window):
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    emb = model.encoder_net(x)
    return emb.cpu().numpy()




def read_eda():
    global latest_eda

    if USE_REAL_DATA:
        return latest_eda

    return 0.6 + random.uniform(-0.08, 0.08)



def main():
    print("Loading encoder...")
    encoder = load_encoder()

    print("Loading classifier...")
    clf = load(os.path.join(MODEL_DIR, CLASSIFIER_FILE))
    scaler = load(os.path.join(MODEL_DIR, SCALER_FILE))

    if USE_REAL_DATA:
        print("Starting MQTT...")
        start_mqtt()
    else:
        print("Using mock EDA data...")

    print("Start inference...")
    print(f"Need {WINDOW_SIZE} points before first prediction.\n")

    buffer = deque(maxlen=WINDOW_SIZE)

    i = 0
    while True:
        try:
            eda = read_eda()

            if eda is None:
                print("Waiting for real EDA data...")
                time.sleep(1)
                continue

            buffer.append(float(eda))

            if len(buffer) < WINDOW_SIZE:
                if len(buffer) % 50 == 0:
                    print(f"Collecting window: {len(buffer)}/{WINDOW_SIZE}")
                time.sleep(SLEEP_TIME)
                continue

            window = np.array(buffer, dtype=np.float32)

            emb = extract_embedding(encoder, window)
            emb_scaled = scaler.transform(emb)

            pred = clf.predict(emb_scaled)[0]
            label = "Stress" if int(pred) == 1 else "No Stress"

            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(emb_scaled)[0]
                stress_prob = prob[1]
                print(f"[{i}] {label} | pred={pred} | stress_prob={stress_prob:.3f}")
                stress_precent = prob[1] * 100
                print(f"Stress Level: {stress_precent:.1f}%")
            else:
                print(f"[{i}] {label} | pred={pred}")

            i += 1
            time.sleep(SLEEP_TIME)

        except KeyboardInterrupt:
            print("\nStopped.")
            break

        except Exception as e:
            print("Error:", e)
            time.sleep(1)


if __name__ == "__main__":
    main()