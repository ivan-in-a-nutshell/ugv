"""
laptop_v1.py  -- runs ON THE LAPTOP. Baseline V1: gimbal driven directly by the latest YOLO-World detection.

Usage:
  python laptop_v1.py --robot 192.168.4.1 --prompt "a red backpack"
  python laptop_v1.py --robot 192.168.4.1 --prompt "a red backpack" --record rec/seq01   (also saves frames for annotation)
  python laptop_v1.py --robot 192.168.4.1 --prompt "a red backpack" --no-control        (detect only, gimbal stays still)

Keys in the window: q = quit, c = re-center gimbal.
"""
import os
os.environ.setdefault("YOLO_OFFLINE", "true")
print("[0/3] importing libraries...", flush=True)
import argparse
import csv
import json
import math
import os
import time

os.environ.setdefault("YOLO_OFFLINE", "true")   # skip Ultralytics' internet check (robot Wi-Fi has no internet)

import cv2
import numpy as np
import zmq
from ultralytics import YOLOWorld

ap = argparse.ArgumentParser()
ap.add_argument("--robot", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--model", default="yolov8s-worldv2.pt")
ap.add_argument("--conf", type=float, default=0.10)
ap.add_argument("--calib", default="calib.npz")
ap.add_argument("--hfov", type=float, default=65.0, help="used only if calib.npz is missing")
ap.add_argument("--deadband", type=float, default=1.0, help="deg; ignore smaller errors")
ap.add_argument("--gain", type=float, default=1.0, help="fraction of the angular error corrected per detection")
ap.add_argument("--pan-sign", type=float, default=1.0, help="flip to -1 if the gimbal turns the wrong way")
ap.add_argument("--tilt-sign", type=float, default=1.0)
ap.add_argument("--record", default=None)
ap.add_argument("--no-control", action="store_true")
ap.add_argument("--log", default="v1_log.csv")
args = ap.parse_args()

# ---------------- camera intrinsics ----------------
calib = None
if os.path.exists(args.calib):
    d = np.load(args.calib)
    K = d["K"]
    calib = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    print("Loaded calibration fx fy cx cy =", calib)

# ---------------- network ----------------
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.CONFLATE, 1)        # keep only the newest frame
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(f"tcp://{args.robot}:5555")
push = ctx.socket(zmq.PUSH)
push.connect(f"tcp://{args.robot}:5556")

# ---------------- detector ----------------
print("[1/3] loading model...", flush=True)
model = YOLOWorld(args.model)
model.set_classes([args.prompt])
print("[2/3] model ready, waiting for frames from", args.robot, flush=True)
first_frame = True

if args.record:
    os.makedirs(args.record, exist_ok=True)

log_f = open(args.log, "w", newline="")
log = csv.writer(log_f)
log.writerow(["frame_id", "t_capture", "det_ms", "found", "u", "v", "conf",
              "err_pan_deg", "err_tilt_deg", "pan_cmd", "tilt_cmd"])

while True:
    msg = sub.recv()
    hdr_bytes, jpg = msg.split(b"\n", 1)
    hdr = json.loads(hdr_bytes)
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    if first_frame:
        print(f"[3/3] receiving frames {w}x{h}, opening window", flush=True)
        first_frame = False

    if args.record:
        with open(os.path.join(args.record, f"{hdr['frame_id']:06d}_{hdr['t_capture']:.6f}.jpg"), "wb") as f:
            f.write(jpg)

    if calib is None:
        fx = fy = (w / 2) / math.tan(math.radians(args.hfov / 2))
        cx, cy = w / 2, h / 2
    else:
        fx, fy, cx, cy = calib

    t0 = time.perf_counter()
    res = model.predict(frame, conf=args.conf, verbose=False)[0]
    det_ms = (time.perf_counter() - t0) * 1000.0

    found = len(res.boxes) > 0
    u = v = conf = err_pan = err_tilt = None
    pan_new, tilt_new = hdr["pan_cmd"], hdr["tilt_cmd"]

    if found:
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        conf = float(res.boxes.conf[i])
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        # pixel error -> angle (pinhole model)
        err_pan = math.degrees(math.atan2(u - cx, fx))
        err_tilt = math.degrees(math.atan2(v - cy, fy))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # V1: new absolute command = gimbal angle at capture time + measured angular error
        if abs(err_pan) > args.deadband:
            pan_new = hdr["pan_cmd"] + args.pan_sign * args.gain * err_pan
        if abs(err_tilt) > args.deadband:
            tilt_new = hdr["tilt_cmd"] - args.tilt_sign * args.gain * err_tilt  # image y grows downward

        if not args.no_control:
            push.send_json({"frame_id": hdr["frame_id"], "t_capture": hdr["t_capture"],
                            "pan": pan_new, "tilt": tilt_new, "spd": 0, "acc": 0})

    log.writerow([hdr["frame_id"], f"{hdr['t_capture']:.6f}", f"{det_ms:.1f}", int(found),
                  u, v, conf, err_pan, err_tilt, pan_new, tilt_new])

    cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(frame, f"det {det_ms:.0f} ms  pan {hdr['pan_cmd']:.1f} tilt {hdr['tilt_cmd']:.1f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("V1 baseline", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("c"):
        push.send_json({"frame_id": hdr["frame_id"], "t_capture": hdr["t_capture"],
                        "pan": 0, "tilt": 0, "spd": 0, "acc": 0})

log_f.close()
cv2.destroyAllWindows()
