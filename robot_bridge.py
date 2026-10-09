"""
robot_bridge.py  -- runs ON THE ROBOT (Raspberry Pi of the UGV Beast).

- Captures camera frames, timestamps them at capture, streams JPEGs to the laptop (ZeroMQ PUB, port 5555).
- Receives pan/tilt commands from the laptop (ZeroMQ PULL, port 5556) and forwards them to the ESP32 as JSON (T:133).
- Logs end-to-end latency (frame capture -> command sent to ESP32) on the Pi's own clock, so no clock sync is needed.

Stop the stock Waveshare app first: it holds the UART and the camera.
"""
import argparse
import csv
import json
import threading
import time

import cv2
import serial
import zmq

ap = argparse.ArgumentParser()
ap.add_argument("--serial", default="/dev/ttyAMA0", help="Pi 5: /dev/ttyAMA0, Pi 4: /dev/serial0")
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--jpeg-quality", type=int, default=80)
ap.add_argument("--log", default="latency_log.csv")
ap.add_argument("--max-speed", type=float, default=0.25, help="hard limit on track speed sent to ESP32")
ap.add_argument("--drive-timeout", type=float, default=1.5, help="s without drive command -> stop tracks")
args = ap.parse_args()

# ---------------- ESP32 serial ----------------
ser = serial.Serial(args.serial, 115200, timeout=0.1)
ser_lock = threading.Lock()


def esp_send(cmd: dict):
    with ser_lock:
        ser.write((json.dumps(cmd) + "\n").encode())


state = {
    "pan_cmd": 0.0, "tilt_cmd": 0.0,   # last commanded angles (always known)
    "pan_fb": None, "tilt_fb": None,   # reported angles, if the firmware sends them
    "fb_keys_printed": False,
}


def serial_reader():
    while True:
        line = ser.readline()
        if not line:
            continue
        try:
            msg = json.loads(line.decode(errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not state["fb_keys_printed"]:
            print("[ESP32 feedback sample]", msg)
            state["fb_keys_printed"] = True
        if "pan" in msg:
            state["pan_fb"] = msg["pan"]
        if "tilt" in msg:
            state["tilt_fb"] = msg["tilt"]


threading.Thread(target=serial_reader, daemon=True).start()

esp_send({"T": 900, "main": 3, "module": 2})   # chassis = UGV Beast, module = pan-tilt
time.sleep(0.2)
esp_send({"T": 131, "cmd": 1})                  # continuous chassis feedback ON
time.sleep(0.2)
esp_send({"T": 133, "X": 0, "Y": 0, "SPD": 0, "ACC": 0})  # center gimbal
time.sleep(1.0)

# ---------------- ZeroMQ ----------------
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.setsockopt(zmq.SNDHWM, 2)
pub.bind("tcp://*:5555")

pull = ctx.socket(zmq.PULL)
pull.bind("tcp://*:5556")

log_f = open(args.log, "w", newline="")
log = csv.writer(log_f)
log.writerow(["frame_id", "t_capture", "t_cmd_sent", "latency_ms", "pan_cmd", "tilt_cmd"])


state["last_drive"] = 0.0
state["driving"] = False


def drive_watchdog():
    # safety: if the laptop stops sending drive commands (crash, Wi-Fi drop), stop the tracks
    while True:
        time.sleep(0.1)
        if state["driving"] and time.monotonic() - state["last_drive"] > args.drive_timeout:
            esp_send({"T": 1, "L": 0, "R": 0})
            state["driving"] = False
            print("[watchdog] no drive command -> STOP")


threading.Thread(target=drive_watchdog, daemon=True).start()


def command_receiver():
    while True:
        cmd = pull.recv_json()
        if cmd.get("type") == "drive":
            L = max(-args.max_speed, min(args.max_speed, float(cmd["L"])))
            R = max(-args.max_speed, min(args.max_speed, float(cmd["R"])))
            esp_send({"T": 1, "L": round(L, 3), "R": round(R, 3)})
            state["last_drive"] = time.monotonic()
            state["driving"] = (L != 0 or R != 0)
            continue
        pan = max(-180.0, min(180.0, float(cmd["pan"])))
        tilt = max(-30.0, min(90.0, float(cmd["tilt"])))
        esp_send({"T": 133, "X": round(pan, 2), "Y": round(tilt, 2),
                  "SPD": int(cmd.get("spd", 0)), "ACC": int(cmd.get("acc", 0))})
        t_sent = time.monotonic()
        state["pan_cmd"], state["tilt_cmd"] = pan, tilt
        lat_ms = (t_sent - cmd["t_capture"]) * 1000.0
        log.writerow([cmd["frame_id"], f"{cmd['t_capture']:.6f}", f"{t_sent:.6f}",
                      f"{lat_ms:.1f}", f"{pan:.2f}", f"{tilt:.2f}"])
        log_f.flush()


threading.Thread(target=command_receiver, daemon=True).start()

# ---------------- camera loop ----------------
cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_id = 0
t_last = time.monotonic()
print("Streaming on tcp://*:5555, commands on tcp://*:5556")
while True:
    ok, frame = cap.read()
    t_cap = time.monotonic()
    if not ok:
        continue
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
    header = {
        "frame_id": frame_id, "t_capture": t_cap,
        "pan_cmd": state["pan_cmd"], "tilt_cmd": state["tilt_cmd"],
        "pan_fb": state["pan_fb"], "tilt_fb": state["tilt_fb"],
    }
    # single-part message: json header + b"\n" + jpeg  (lets the laptop use CONFLATE = always newest frame)
    pub.send(json.dumps(header).encode() + b"\n" + jpg.tobytes())
    frame_id += 1
    if frame_id % 100 == 0:
        now = time.monotonic()
        print(f"camera fps: {100 / (now - t_last):.1f}")
        t_last = now
