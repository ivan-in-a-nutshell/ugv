"""
replay_bridge.py -- stands in for robot_bridge.py on the laptop: streams a recorded sequence in the same message
format (json header + b"\\n" + jpeg on tcp 5555) and accepts commands on tcp 5556. Gimbal commands are echoed
back in the header (pan_cmd / tilt_cmd) like the real bridge, drive commands are only counted.
The image does NOT react to the commands, so this tests that the laptop scripts run end to end, not control.

  python replay_bridge.py --seq rec/S1_seq04 --loop
  python laptop_approach.py --robot 127.0.0.1 --prompt "a yellow chair" --no-drive --headless --duration 20
"""
import argparse
import glob
import json
import os
import threading
import time

import zmq

ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--loop", action="store_true")
args = ap.parse_args()

frames = sorted(glob.glob(os.path.join(args.seq, "frames", "*.jpg")))
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.setsockopt(zmq.SNDHWM, 2)
pub.bind("tcp://*:5555")
pull = ctx.socket(zmq.PULL)
pull.bind("tcp://*:5556")
state = {"pan_cmd": 0.0, "tilt_cmd": 0.0, "n_gimbal": 0, "n_drive": 0}


def receiver():
    while True:
        cmd = pull.recv_json()
        if cmd.get("type") == "drive":
            state["n_drive"] += 1
            continue
        state["pan_cmd"] = max(-180.0, min(180.0, float(cmd["pan"])))
        state["tilt_cmd"] = max(-30.0, min(90.0, float(cmd["tilt"])))
        state["n_gimbal"] += 1


threading.Thread(target=receiver, daemon=True).start()
print(f"replaying {len(frames)} frames from {args.seq}")
frame_id = 0
while True:
    for f in frames:
        t = time.monotonic()
        hdr = {"frame_id": frame_id, "t_capture": t, "pan_cmd": state["pan_cmd"], "tilt_cmd": state["tilt_cmd"],
               "pan_fb": None, "tilt_fb": None}
        pub.send(json.dumps(hdr).encode() + b"\n" + open(f, "rb").read())
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"frame {frame_id}  gimbal cmds {state['n_gimbal']}  drive cmds {state['n_drive']}", flush=True)
        time.sleep(max(0.0, 1.0 / args.fps - (time.monotonic() - t)))
    if not args.loop:
        break
