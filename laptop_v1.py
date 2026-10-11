"""
laptop_v1.py  -- runs ON THE LAPTOP. Baseline V1: gimbal driven directly by the latest YOLO-World detection.

Usage:
  python laptop_v1.py --robot 192.168.4.1 --prompt "a red backpack"
  python laptop_v1.py --robot 192.168.4.1 --prompt "a red backpack" --record rec/seq01   (also saves frames for annotation)
  python laptop_v1.py --robot 192.168.4.1 --prompt "a red backpack" --no-control        (detect only, gimbal stays still)

Keys (click the video window first so it has keyboard focus):
  q / ESC = quit, c = re-center gimbal and pause tracking, SPACE = pause / resume tracking.
  Ctrl+C in the terminal also quits cleanly.

Settle gating: the image only shows the result of a gimbal move after the command reached the bridge, the
servos finished moving and a new frame was captured (~0.3-0.5 s). Acting on frames from before that adds the
same error again and again -> overshoot and oscillation. So after each move we ignore detections until a frame
is captured with our command already applied (hdr pan_cmd/tilt_cmd == what we sent) plus a settle time.
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
ap.add_argument("--gain", type=float, default=0.8, help="fraction of the angular error corrected per detection")
ap.add_argument("--settle", type=float, default=0.15, help="s; extra wait after a move before trusting detections")
ap.add_argument("--settle-dps", type=float, default=200.0, help="assumed gimbal speed, deg/s (adds |move|/this)")
ap.add_argument("--settle-timeout", type=float, default=1.5, help="s; give up waiting for the move to show up")
ap.add_argument("--no-settle", action="store_true", help="old behaviour: act on every detection (oscillates)")
ap.add_argument("--pan-sign", type=float, default=1.0, help="flip to -1 if the gimbal turns the wrong way")
ap.add_argument("--tilt-sign", type=float, default=1.0)
ap.add_argument("--record", default=None)
ap.add_argument("--no-control", action="store_true")
ap.add_argument("--log", default="v1_log.csv")
args = ap.parse_args()

# ---------------- camera intrinsics ----------------
calib = None
K = dist = None
if os.path.exists(args.calib):
    d = np.load(args.calib)
    K, dist = d["K"], d["dist"]
    calib = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    print("Loaded calibration fx fy cx cy =", calib, " dist =", dist.ravel())


def pixel_to_angles(u, v, fx, fy, cx, cy):
    """Angles (deg) of pixel (u, v) from the optical axis. With calibration, lens distortion is removed first:
    the lens is ~100 deg wide with strong barrel distortion, so the plain pinhole formula is off near the edges."""
    if K is not None:
        x, y = cv2.undistortPoints(np.array([[[u, v]]], np.float64), K, dist)[0, 0]
    else:
        x, y = (u - cx) / fx, (v - cy) / fy
    return math.degrees(math.atan(x)), math.degrees(math.atan(y))

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
              "err_pan_deg", "err_tilt_deg", "pan_cmd", "tilt_cmd", "moving"])


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def send_cmd(hdr, pan, tilt):
    # clamp like the bridge does, so the value echoed back in the frame header matches exactly
    pan, tilt = clamp(pan, -180.0, 180.0), clamp(tilt, -30.0, 90.0)
    push.send_json({"frame_id": hdr["frame_id"], "t_capture": hdr["t_capture"],
                    "pan": pan, "tilt": tilt, "spd": 0, "acc": 0})
    move = max(abs(pan - hdr["pan_cmd"]), abs(tilt - hdr["tilt_cmd"]))
    return {"pan": pan, "tilt": tilt, "settle": args.settle + move / args.settle_dps,
            "t_sent": time.monotonic(), "t_applied": None}


pending = None    # last move we are waiting for (None = gimbal assumed still, detections usable)
paused = False

while True:
    try:
        msg = sub.recv()
    except KeyboardInterrupt:
        break
    hdr_bytes, jpg = msg.split(b"\n", 1)
    hdr = json.loads(hdr_bytes)

    # ---- settle gating: is this frame captured after our last move finished? ----
    if pending is not None:
        if pending["t_applied"] is None and abs(hdr["pan_cmd"] - pending["pan"]) < 0.05 \
                and abs(hdr["tilt_cmd"] - pending["tilt"]) < 0.05:
            pending["t_applied"] = hdr["t_capture"]   # first frame captured after the bridge applied the move
        if pending["t_applied"] is not None and hdr["t_capture"] - pending["t_applied"] >= pending["settle"]:
            pending = None
        elif time.monotonic() - pending["t_sent"] > args.settle_timeout:
            pending = None                             # command lost or overwritten -- don't stay stuck
    moving = pending is not None
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
        err_pan, err_tilt = pixel_to_angles(u, v, fx, fy, cx, cy)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # V1: new absolute command = gimbal angle at capture time + measured angular error
        if abs(err_pan) > args.deadband:
            pan_new = hdr["pan_cmd"] + args.pan_sign * args.gain * err_pan
        if abs(err_tilt) > args.deadband:
            tilt_new = hdr["tilt_cmd"] - args.tilt_sign * args.gain * err_tilt  # image y grows downward

        changed = pan_new != hdr["pan_cmd"] or tilt_new != hdr["tilt_cmd"]
        if not args.no_control and not paused and changed and (args.no_settle or not moving):
            pending = send_cmd(hdr, pan_new, tilt_new)
            if args.no_settle:
                pending = None

    log.writerow([hdr["frame_id"], f"{hdr['t_capture']:.6f}", f"{det_ms:.1f}", int(found),
                  u, v, conf, err_pan, err_tilt, pan_new, tilt_new, int(moving)])

    cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    status = "PAUSED (SPACE)" if paused else ("moving" if moving else "tracking")
    cv2.putText(frame, f"det {det_ms:.0f} ms  pan {hdr['pan_cmd']:.1f} tilt {hdr['tilt_cmd']:.1f}  {status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("V1 baseline", frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), ord("Q"), 27):          # 27 = ESC
        break
    if k in (ord("c"), ord("C")):
        # pause too, otherwise the next detection immediately turns the gimbal back to the target
        paused = True
        pending = send_cmd(hdr, 0.0, 0.0)
        print("re-centered, tracking PAUSED (SPACE to resume)", flush=True)
    if k == ord(" "):
        paused = not paused
        print("tracking PAUSED" if paused else "tracking RESUMED", flush=True)

log_f.close()
cv2.destroyAllWindows()
