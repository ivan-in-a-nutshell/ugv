"""
record.py  -- runs ON THE LAPTOP. Records EVERY camera frame from the robot for annotation and replay.

laptop_v1.py --record only saves the frames it ran detection on (~1 in 3). The open-loop study replays a sequence
with the detector throttled to 1/2/5/10 fps, so it needs the full camera-rate stream. This script saves the JPEGs
exactly as the bridge sent them (no re-encoding) plus a per-frame CSV with capture time and gimbal angles.
It only listens, so it can run next to laptop_v1.py: one run = closed-loop result + full recording.

Usage:
  python record.py --robot 192.168.50.5 --name S1_seq01 --scene lab1 --prompt "a red backpack"
  python record.py --robot 192.168.50.5 --name S4_seq02 --scene corridor --prompt "a bottle" --duration 20

Output (rec/ is git-ignored):
  rec/<name>/frames/<frame_id>.jpg
  rec/<name>/frames.csv   frame_id, t_capture (Pi clock), t_recv (laptop clock), pan_cmd, tilt_cmd, pan_fb, tilt_fb
  rec/<name>/meta.json    scenario, scene, prompt, notes, counts

Keys (click the preview window first): q / ESC = stop. Ctrl+C in the terminal also stops cleanly.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import zmq

ap = argparse.ArgumentParser()
ap.add_argument("--robot", required=True)
ap.add_argument("--name", required=True, help="sequence name, e.g. S1_seq01 (scenario prefix S1..S6)")
ap.add_argument("--scene", required=True, help="where it was filmed, e.g. lab1 -- used to split val/test by scene")
ap.add_argument("--prompt", default="", help="text prompt of the target object")
ap.add_argument("--notes", default="")
ap.add_argument("--out", default="rec")
ap.add_argument("--duration", type=float, default=None, help="s; stop automatically (default: until q / Ctrl+C)")
ap.add_argument("--no-preview", action="store_true")
args = ap.parse_args()

seq_dir = os.path.join(args.out, args.name)
if os.path.exists(seq_dir):
    sys.exit(f"{seq_dir} already exists -- pick another --name (never overwrite a recording)")
os.makedirs(os.path.join(seq_dir, "frames"))

# no CONFLATE here: we want every frame, so keep a deep receive queue instead
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.RCVHWM, 1000)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.setsockopt(zmq.RCVTIMEO, 3000)
sub.connect(f"tcp://{args.robot}:5555")

csv_f = open(os.path.join(seq_dir, "frames.csv"), "w", newline="")
log = csv.writer(csv_f)
log.writerow(["frame_id", "t_capture", "t_recv", "pan_cmd", "tilt_cmd", "pan_fb", "tilt_fb"])

meta = {"name": args.name, "scenario": args.name.split("_")[0], "scene": args.scene, "prompt": args.prompt,
        "notes": args.notes, "robot": args.robot, "started": datetime.now().isoformat(timespec="seconds")}

print(f"waiting for frames from {args.robot} ...", flush=True)
n = dropped = 0
first_id = last_id = None
t_start = None
t_status = time.monotonic()
try:
    while True:
        try:
            msg = sub.recv()
        except zmq.Again:
            print("no frame for 3 s -- is robot_bridge.py running?", flush=True)
            continue
        t_recv = time.monotonic()
        hdr_bytes, jpg = msg.split(b"\n", 1)
        hdr = json.loads(hdr_bytes)
        fid = hdr["frame_id"]
        if first_id is None:
            first_id, t_start = fid, t_recv
            print(f"RECORDING {seq_dir}", flush=True)
        elif fid != last_id + 1:
            dropped += max(0, fid - last_id - 1)   # frames the bridge sent while we were not keeping up
        last_id = fid
        n += 1

        with open(os.path.join(seq_dir, "frames", f"{fid:06d}.jpg"), "wb") as f:
            f.write(jpg)
        log.writerow([fid, f"{hdr['t_capture']:.6f}", f"{t_recv:.6f}", hdr["pan_cmd"], hdr["tilt_cmd"],
                      hdr["pan_fb"], hdr["tilt_fb"]])

        elapsed = t_recv - t_start
        if t_recv - t_status >= 1.0:
            print(f"  {elapsed:5.1f} s  {n} frames  ({n / max(elapsed, 1e-6):.1f} fps)  dropped {dropped}", flush=True)
            t_status = t_recv

        if not args.no_preview:
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {args.name}  {elapsed:.1f} s  dropped {dropped}", (35, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("record", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
        if args.duration is not None and elapsed >= args.duration:
            break
except KeyboardInterrupt:
    pass

csv_f.close()
cv2.destroyAllWindows()
duration = (time.monotonic() - t_start) if t_start is not None else 0.0
meta.update({"frames": n, "dropped": dropped, "first_frame_id": first_id, "last_frame_id": last_id,
             "duration_s": round(duration, 2), "fps": round(n / duration, 2) if duration else 0.0})
with open(os.path.join(seq_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"saved {n} frames ({duration:.1f} s, {meta['fps']} fps, dropped {dropped}) to {seq_dir}")
