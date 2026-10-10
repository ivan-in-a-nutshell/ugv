"""
laptop_approach.py  -- runs ON THE LAPTOP. Search for a text-described object, lock the gimbal on it,
turn the chassis toward it and drive up to it.

Needs the updated robot_bridge.py on the Pi (drive commands + stop watchdog).

  python laptop_approach.py --robot 192.168.50.5 --prompt "a red backpack" --no-drive
  python laptop_approach.py --robot 192.168.50.5 --prompt "a red backpack"

Keys (click the video window first so it has keyboard focus):
  SPACE = pause/resume driving (emergency stop), q / ESC = stop and quit,
  c = stop tracks, re-center gimbal and HOLD (no tracking/search) until SPACE.
  Ctrl+C in the terminal also stops the tracks and quits.

v2 changes (fixes for "gimbal too fast" and "loses target while the chassis turns"):
  - gimbal speed/acceleration limits (SPD/ACC sent to the ESP32) + max angle change per update
  - yaw feed-forward: while the chassis rotates, the gimbal is counter-rotated by the estimated chassis
    rotation, so the camera keeps pointing at the same spot in the world instead of being dragged with the body
  - gentler chassis turning: turn deadband, turn-rate limit, turn slows down near alignment
  - local re-acquisition: when the target is lost, look first around where it was last seen
    (corrected for chassis rotation) before falling back to the full sweep

v3: settle gating (same fix as laptop_v1.py). A gimbal correction is made only from a frame captured after
the previous move was applied and the gimbal had time to settle; otherwise the same error is corrected
several times while the move is still in flight -> overshoot/oscillation. Every command we send gets an
index; the frame header echoes the angles the bridge had applied at capture, so matching it against the
recent send history tells us which command the frame already shows. Yaw feed-forward still runs every frame.
"""
import os
os.environ.setdefault("YOLO_OFFLINE", "true")
print("[0/3] importing libraries...", flush=True)
import argparse
import csv
import json
import math
import signal
import time
from collections import deque

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
ap.add_argument("--hfov", type=float, default=65.0)
# gimbal
ap.add_argument("--pan-sign", type=float, default=1.0)
ap.add_argument("--tilt-sign", type=float, default=1.0)
ap.add_argument("--gain", type=float, default=0.8,
                help="fraction of the measured angular error corrected per move (2.0 also worked uncalibrated)")
ap.add_argument("--deadband", type=float, default=1.5, help="deg")
ap.add_argument("--max-step", type=float, default=100.0, help="max gimbal angle change per update, deg")
ap.add_argument("--gimbal-spd", type=int, default=0, help="ESP32 gimbal speed (0 = fastest; lower = slower)")
ap.add_argument("--gimbal-acc", type=int, default=0, help="ESP32 gimbal acceleration (0 = fastest)")
ap.add_argument("--settle", type=float, default=0.15, help="s; extra wait after a move before trusting detections")
ap.add_argument("--settle-dps", type=float, default=200.0,
                help="assumed gimbal speed at --gimbal-spd, deg/s (adds |move|/this); lower if it still overshoots")
ap.add_argument("--settle-timeout", type=float, default=1.5, help="s; give up waiting for the move to show up")
ap.add_argument("--no-settle", action="store_true", help="old behaviour: correct on every detection")
# chassis
ap.add_argument("--drive-sign", type=float, default=1.0)
ap.add_argument("--turn-sign", type=float, default=1.0)
ap.add_argument("--max-speed", type=float, default=0.12, help="m/s forward")
ap.add_argument("--k-turn", type=float, default=0.002, help="track speed difference per degree of pan")
ap.add_argument("--max-turn", type=float, default=0.06, help="max track speed difference for turning")
ap.add_argument("--turn-deadband", type=float, default=5.0, help="don't turn the chassis if |pan| < this, deg")
ap.add_argument("--align-deg", type=float, default=12.0, help="drive forward only when |pan| < this")
ap.add_argument("--arrive-ratio", type=float, default=0.40)
# yaw feed-forward
ap.add_argument("--track-width", type=float, default=0.20, help="m, distance between track centers")
ap.add_argument("--yaw-comp", type=float, default=1.0,
                help="gain of gimbal counter-rotation (0 = off; increase if target still drifts out while turning)")
# search
ap.add_argument("--lost-timeout", type=float, default=1.0)
ap.add_argument("--local-offsets", default="0,-25,25,-50,50", help="deg around last seen pan for local search")
ap.add_argument("--sweep", default="-90,-45,0,45,90")
ap.add_argument("--sweep-dwell", type=float, default=1.5)
ap.add_argument("--spin-time", type=float, default=1.5)
ap.add_argument("--spin-speed", type=float, default=0.08)
ap.add_argument("--no-drive", action="store_true")
ap.add_argument("--log", default="approach_log.csv")
args = ap.parse_args()

GLOBAL_SWEEP = [float(a) for a in args.sweep.split(",")]
LOCAL_OFFSETS = [float(a) for a in args.local_offsets.split(",")]

calib = None
if os.path.exists(args.calib):
    K = np.load(args.calib)["K"]
    calib = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    print("Loaded calibration fx fy cx cy =", calib)

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.CONFLATE, 1)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(f"tcp://{args.robot}:5555")
push = ctx.socket(zmq.PUSH)
push.connect(f"tcp://{args.robot}:5556")

print("[1/3] loading model...", flush=True)
model = YOLOWorld(args.model)
model.set_classes([args.prompt])
print("[2/3] model ready, waiting for frames from", args.robot, flush=True)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


paused = False
hold = False                          # after 'c': gimbal centered, no tracking/search until SPACE
gimbal_pan, gimbal_tilt = 0.0, 0.0   # what we last commanded (our own copy, includes feed-forward)
turn_now = 0.0                        # last turn command sent, "toward positive pan" units
cmd_idx = 0                           # index of the last gimbal command sent
sent_hist = deque(maxlen=100)         # (idx, pan, tilt) of recent gimbal commands
pending = None                        # move we wait for before trusting detections (None = settled)


def send_gimbal(hdr, pan, tilt, limit_step=True, correction=True):
    """correction=False: feed-forward only, never starts a new settle wait."""
    global gimbal_pan, gimbal_tilt, cmd_idx, pending
    if limit_step:
        pan = gimbal_pan + clamp(pan - gimbal_pan, -args.max_step, args.max_step)
        tilt = gimbal_tilt + clamp(tilt - gimbal_tilt, -args.max_step, args.max_step)
    pan, tilt = clamp(pan, -180.0, 180.0), clamp(tilt, -30.0, 90.0)
    changed = abs(pan - gimbal_pan) > 0.01 or abs(tilt - gimbal_tilt) > 0.01
    gimbal_pan, gimbal_tilt = pan, tilt
    push.send_json({"frame_id": hdr["frame_id"], "t_capture": hdr["t_capture"],
                    "pan": gimbal_pan, "tilt": gimbal_tilt,
                    "spd": args.gimbal_spd, "acc": args.gimbal_acc})
    cmd_idx += 1
    sent_hist.append((cmd_idx, gimbal_pan, gimbal_tilt))
    if correction and changed and not args.no_settle:
        move = max(abs(gimbal_pan - hdr["pan_cmd"]), abs(gimbal_tilt - hdr["tilt_cmd"]))
        pending = {"idx": cmd_idx, "settle": args.settle + move / args.settle_dps,
                   "t_sent": time.monotonic(), "t_applied": None}


def update_settle(hdr):
    """True while the frame may not yet show the result of the last correction move."""
    global pending
    if pending is None:
        return False
    if pending["t_applied"] is None:
        # newest command whose angles the bridge had applied when this frame was captured
        shown = next((i for i, p, t in reversed(sent_hist)
                      if abs(p - hdr["pan_cmd"]) < 1e-6 and abs(t - hdr["tilt_cmd"]) < 1e-6), None)
        if shown is not None and shown >= pending["idx"]:
            pending["t_applied"] = hdr["t_capture"]
    if pending["t_applied"] is not None and hdr["t_capture"] - pending["t_applied"] >= pending["settle"]:
        pending = None
    elif time.monotonic() - pending["t_sent"] > args.settle_timeout:
        pending = None                   # command lost -- don't stay stuck
    return pending is not None


def send_drive(v, turn):
    """v = forward speed; turn > 0 rotates the robot toward positive pan (after --turn-sign is set right)."""
    global turn_now
    if args.no_drive or paused:
        v, turn = 0.0, 0.0
    turn = clamp(turn, -args.max_turn, args.max_turn)
    turn_now = turn
    w = args.turn_sign * turn
    L = args.drive_sign * (v + w)
    R = args.drive_sign * (v - w)
    push.send_json({"type": "drive", "L": L, "R": R})
    return L, R


def chassis_yaw_rate_deg():
    # rotation of the body toward positive pan, deg/s, estimated from the commanded track speeds
    return math.degrees(2.0 * turn_now / args.track_width)


log_f = open(args.log, "w", newline="")
log = csv.writer(log_f)
log.writerow(["frame_id", "t_capture", "state", "det_ms", "found", "box_ratio",
              "pan_cmd", "tilt_cmd", "yaw_ff_deg", "L", "R", "moving"])

state = "SEARCH"
last_seen = 0.0
last_pan_seen = 0.0          # gimbal pan (relative to body) where the target was last seen
search_list = list(GLOBAL_SWEEP)
search_is_local = False
search_idx = 0
search_t0 = time.time()
spin_until = 0.0
first_frame = True
t_prev = time.time()

stop_requested = False


def on_sigint(*_):
    # first Ctrl+C: leave the loop and send the stop commands below; second Ctrl+C: hard exit
    global stop_requested
    stop_requested = True
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print("Ctrl+C -> stopping tracks and quitting", flush=True)


signal.signal(signal.SIGINT, on_sigint)

while not stop_requested:
    msg = sub.recv()
    hdr_bytes, jpg = msg.split(b"\n", 1)
    hdr = json.loads(hdr_bytes)
    moving = update_settle(hdr)
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    h, w_img = frame.shape[:2]
    if first_frame:
        print(f"[3/3] receiving frames {w_img}x{h}, opening window", flush=True)
        first_frame = False

    if calib is None:
        fx = fy = (w_img / 2) / math.tan(math.radians(args.hfov / 2))
        cx, cy = w_img / 2, h / 2
    else:
        fx, fy, cx, cy = calib

    t0 = time.perf_counter()
    res = model.predict(frame, conf=args.conf, verbose=False)[0]
    det_ms = (time.perf_counter() - t0) * 1000.0
    now = time.time()
    dt = now - t_prev
    t_prev = now

    # ---- yaw feed-forward: how much did the body rotate since the last update? ----
    yaw_ff = args.yaw_comp * chassis_yaw_rate_deg() * dt
    last_pan_seen -= yaw_ff      # the target's bearing relative to the body shrinks as we turn toward it

    found = len(res.boxes) > 0
    box_ratio = None
    L = R = 0.0

    if hold:
        state = "HOLD"
        L, R = send_drive(0.0, 0.0)

    elif found:
        last_seen = now
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        box_ratio = (y2 - y1) / h
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        err_pan = math.degrees(math.atan2(u - cx, fx))
        err_tilt = math.degrees(math.atan2(v - cy, fy))

        # bearing of the target relative to the body at capture time, then corrected for rotation since
        target_pan = hdr["pan_cmd"] + args.pan_sign * err_pan - yaw_ff
        target_tilt = hdr["tilt_cmd"] - args.tilt_sign * err_tilt
        last_pan_seen = target_pan

        pan_new = gimbal_pan - yaw_ff
        tilt_new = gimbal_tilt
        if moving:
            # last correction not visible yet: only keep counter-rotating against the chassis turn
            if yaw_ff != 0.0:
                send_gimbal(hdr, pan_new, tilt_new, limit_step=False, correction=False)
        else:
            if abs(err_pan) > args.deadband:
                pan_new = pan_new + args.gain * (target_pan - pan_new)
            if abs(err_tilt) > args.deadband:
                tilt_new = tilt_new + args.gain * (target_tilt - tilt_new)
            send_gimbal(hdr, pan_new, tilt_new)

        if box_ratio >= args.arrive_ratio:
            state = "ARRIVED"
            L, R = send_drive(0.0, 0.0)
        else:
            state = "TRACK"
            bearing = target_pan
            if abs(bearing) < args.turn_deadband:
                turn = 0.0
            else:
                turn = args.k_turn * bearing
            if abs(bearing) < args.align_deg:
                closeness = box_ratio / args.arrive_ratio
                fwd = args.max_speed * max(0.3, 1.0 - closeness)
            else:
                fwd = 0.0
            L, R = send_drive(fwd, turn)
        search_is_local = False

    elif now - last_seen < args.lost_timeout:
        # just lost: stop the body, keep the camera pointed where the target was (counter-rotated)
        state = "LOST"
        L, R = send_drive(0.0, 0.0)
        send_gimbal(hdr, last_pan_seen, gimbal_tilt)

    else:
        state = "SEARCH"
        if not search_is_local and last_seen > 0 and now - last_seen < args.lost_timeout + 0.5:
            # start a local search around the last known bearing
            search_list = [clamp(last_pan_seen + o, -180.0, 180.0) for o in LOCAL_OFFSETS]
            search_is_local = True
            search_idx = 0
            search_t0 = now

        if now < spin_until:
            send_gimbal(hdr, 0.0, 0.0)
            L, R = send_drive(0.0, args.spin_speed)
        else:
            if now - search_t0 > args.sweep_dwell:
                search_idx += 1
                search_t0 = now
                if search_idx >= len(search_list):
                    search_idx = 0
                    if search_is_local:
                        search_list = list(GLOBAL_SWEEP)   # local search failed -> full sweep
                        last_seen = 0.0
                    else:
                        spin_until = now + args.spin_time   # full sweep failed -> rotate body
            send_gimbal(hdr, search_list[search_idx], 0.0, limit_step=False)
            L, R = send_drive(0.0, 0.0)

    log.writerow([hdr["frame_id"], f"{hdr['t_capture']:.6f}", state, f"{det_ms:.1f}", int(found),
                  box_ratio, f"{gimbal_pan:.2f}", f"{gimbal_tilt:.2f}", f"{yaw_ff:.2f}", f"{L:.3f}", f"{R:.3f}", int(moving)])

    cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    color = {"SEARCH": (0, 200, 255), "TRACK": (0, 255, 0), "LOST": (0, 0, 255), "ARRIVED": (255, 0, 255),
             "HOLD": (200, 200, 200)}[state]
    txt = f"{state}{' PAUSED' if paused else ''}{' moving' if moving else ''}  det {det_ms:.0f}ms  pan {gimbal_pan:.0f}  L {L:.2f} R {R:.2f}"
    if box_ratio is not None:
        txt += f"  box {box_ratio:.2f}/{args.arrive_ratio:.2f}"
    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imshow("search and approach", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        if hold:
            hold = paused = False         # leave HOLD: resume tracking and driving
        else:
            paused = not paused
        send_drive(0.0, 0.0)
        print("PAUSED" if paused else "RESUMED", flush=True)
    if k in (ord("c"), ord("C")):
        # hold, otherwise the next detection / search step immediately moves the gimbal away again
        hold = paused = True
        send_drive(0.0, 0.0)
        send_gimbal(hdr, 0.0, 0.0, limit_step=False)
        print("re-centered, HOLD (SPACE to resume)", flush=True)
    if k in (ord("q"), ord("Q"), 27):     # 27 = ESC
        break

for _ in range(5):
    push.send_json({"type": "drive", "L": 0, "R": 0})
    time.sleep(0.05)
log_f.close()
cv2.destroyAllWindows()
