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

v4 (midterm fixes):
  - camera jumps: the bridge echoes a new pan_cmd as soon as it forwards the command, but the servo needs
    ~0.3-0.4 s to get there. Frames in between were labelled with the new angle while still showing the old view,
    so "pan_cmd + error" counted the error twice. The gimbal correction was already gated, but last_pan_seen and
    the chassis bearing were still updated from those frames, and one missed detection sent the gimbal to that
    wrong bearing at full speed (logged: -96 -> -151 deg). Now the bearing is only taken from settled frames,
    a short miss (--miss-grace) keeps everything as it is, LOST holds the gimbal, and --max-step is 35 deg.
  - with several boxes, keep the one closest to the previous box instead of the most confident one.
  - arriving: the old rule (box height >= 40% of the image) stopped ~1.2 m from a chair and metres from a person.
    Now the distance to the object's base on the floor is estimated from the camera height, the gimbal tilt and
    the calibrated angle of the box's bottom edge (d = cam_height / tan(angle below horizon)); stop at
    --arrive-dist, slow down over --slow-dist. The box-height rule stays as a safety stop (--arrive-ratio 0.85).
  - detection via detector.py: zoomed 2x2 tiles while searching and a zoomed crop around a small target while
    tracking (far objects); optional CLIP attribute check (--attr-check, experimental).

v5 (robot test 2026-10-24: "finds it but keeps rotating"):
  - a target found in a frame taken mid-sweep did not update the remembered direction, so the local search
    restarted at the old angle and swept away from it again. Now the area where it was seen is remembered
    (area_pan; the sweep's destination if the frame was mid-move) and searched first, --local-rounds times, and
    again after every full sweep.
  - timeouts in seconds were shorter than one tiles frame (0.7 s): one miss = LOST. Now counted in frames
    (--miss-frames, --lost-frames), and after a miss the next frame uses zoomed tiles again.
  - slower, stop-and-look search: 30 deg steps (20 deg locally) in snake order, move on only after
    --look-frames sharp frames at a position; slower chassis spin.

v6 (robot runs 2026-10-24: drove 4 m to a red pouf, lost it up close):
  - the detector found the pouf in only ~half the frames (score around 0.10). Once locked, weaker boxes
    (--lock-conf) next to the last one count too, and a template tracker (tracker.py) bridges up to --trk-frames
    missed frames (cyan box on screen).
  - on a miss the body kept turning with the last command, at ~1 frame/s (tiles) -> over-turned. Now it stops
    and waits; tiles only in SEARCH (tracking stays at 3-9 frames/s); --max-turn 0.06 -> 0.035.
  - run07: arrived (0.45 m), but below ~1 m the detector hardly fired and the 5-frame tracker budget ran out
    while the tracker still had the pouf -> 6 needless LOST/SEARCH. Close targets get --trk-frames-close 25.
  - run08: the pouf was found almost only by the zoomed tiles (14 detector hits in 142 frames). Tiles again while
    stopped after a miss, and the zoomed ROI for targets up to --roi-px 130 (was 60).

v7 (run10: asked for "a blue chair", locked onto the red pouf): strict first lock, relaxed tracking.
  - first lock (and re-acquisition after LOST): score >= --acquire-conf, CLIP colour check for hue words
    (--colour-min; red/blue/yellow... only), and the same object in --confirm-frames of --confirm-window frames
    while the robot stands still (state CONFIRM, orange box).
  - once locked: as before (--lock-conf next to the target, tracker), and boxes elsewhere are ignored -- it can
    no longer jump to another object while tracking; a new object only through the strict first lock.
  - run11: correct blue chair, then a box 5x smaller next to it counted as 'near' (centre within 200 px) and the
    lock moved to a white box, then the pouf. 'Same object' now needs overlap, or closeness relative to the box
    size plus similar size (weak detections must overlap); and the colour is re-checked every --recheck-every
    frames while tracking, dropping the lock after --recheck-fails failures.
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

from detector import TextDetector, iou
from tracker import TemplateTracker

ap = argparse.ArgumentParser()
ap.add_argument("--robot", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--model", default="yolov8s-worldv2.pt")
ap.add_argument("--conf", type=float, default=0.10)
ap.add_argument("--lock-conf", type=float, default=0.05,
                help="once locked, also accept weaker detections (down to this) next to the last box")
# strict first lock (relaxed only once locked): run10 locked onto a red pouf when asked for "a blue chair"
ap.add_argument("--acquire-conf", type=float, default=0.15, help="detector score needed for a first lock")
ap.add_argument("--colour-min", type=float, default=0.15,
                help="first lock: min CLIP probability of the prompt's colour (only for hue words: red, blue, ...)")
ap.add_argument("--no-colour-check", action="store_true")
ap.add_argument("--confirm-frames", type=int, default=3,
                help="first lock: the candidate must pass in this many frames (robot standing still) ...")
ap.add_argument("--confirm-window", type=int, default=6, help="... out of this many, else it is rejected")
ap.add_argument("--recheck-every", type=int, default=5,
                help="while tracking a prompt with a hue word, re-check the colour every N frames (0 = off) ...")
ap.add_argument("--recheck-min", type=float, default=0.08, help="... min colour probability (lower than the first lock)")
ap.add_argument("--recheck-fails", type=int, default=3, help="... drop the lock after this many failed checks in a row")
ap.add_argument("--trk-frames", type=int, default=5,
                help="max frames in a row bridged by the template tracker when the detector misses (0 = off)")
ap.add_argument("--trk-score", type=float, default=0.6, help="min template match score (normalised correlation)")
ap.add_argument("--trk-frames-close", type=int, default=25,
                help="tracker budget when the target is close (box >= --close-ratio of the image height): up close "
                     "the detector often stops recognising the object, while the tracker can hardly drift")
ap.add_argument("--close-ratio", type=float, default=0.18)
ap.add_argument("--attr-check", action="store_true",
                help="experimental: detect the bare noun, keep boxes CLIP finds closer to the full prompt than to the "
                     "noun (helps 'a yellow chair' vs a red one, removes the target for 'a dark box'; see detector.py)")
ap.add_argument("--attr-conf", type=float, default=0.05, help="noun-stage threshold with --attr-check")
ap.add_argument("--noun", default=None, help="noun for --attr-check, default = last word of the prompt")
ap.add_argument("--no-tiles", action="store_true", help="don't run zoomed tiles while searching")
ap.add_argument("--roi-px", type=float, default=130.0,
                help="while tracking, also run a 2x zoomed crop around the target if its box is lower than this, px")
ap.add_argument("--assoc-px", type=float, default=200.0,
                help="pick the box nearest the previous one if it is within this distance, px")
ap.add_argument("--calib", default="calib.npz")
ap.add_argument("--hfov", type=float, default=65.0)
# gimbal
ap.add_argument("--pan-sign", type=float, default=1.0)
ap.add_argument("--tilt-sign", type=float, default=1.0)
ap.add_argument("--gain", type=float, default=0.8,
                help="fraction of the measured angular error corrected per move (2.0 also worked uncalibrated)")
ap.add_argument("--deadband", type=float, default=1.5, help="deg")
ap.add_argument("--max-step", type=float, default=35.0, help="max gimbal angle change per update, deg")
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
ap.add_argument("--max-turn", type=float, default=0.035, help="max track speed difference for turning")
ap.add_argument("--turn-deadband", type=float, default=5.0, help="don't turn the chassis if |pan| < this, deg")
ap.add_argument("--align-deg", type=float, default=12.0, help="drive forward only when |pan| < this")
ap.add_argument("--arrive-dist", type=float, default=0.45,
                help="m; stop when the object's base on the floor is this close to the camera (ground-plane estimate)")
ap.add_argument("--cam-height", type=float, default=0.25,
                help="m; height of the camera lens above the floor -- MEASURE IT, the distance scales with it")
ap.add_argument("--slow-dist", type=float, default=1.0, help="m; slow down over this distance before the stop point")
ap.add_argument("--arrive-ratio", type=float, default=0.85,
                help="safety stop: box height / image height >= this (objects not standing on the floor)")
# yaw feed-forward
ap.add_argument("--track-width", type=float, default=0.20, help="m, distance between track centers")
ap.add_argument("--yaw-comp", type=float, default=1.0,
                help="gain of gimbal counter-rotation (0 = off; increase if target still drifts out while turning)")
# search
# losing / searching -- counted in detector frames, not seconds: one frame takes 0.1 s (full) to 0.7 s (tiles)
ap.add_argument("--miss-frames", type=int, default=3,
                help="up to this many frames without the target are ignored (gimbal and tracks keep going)")
ap.add_argument("--lost-frames", type=int, default=6,
                help="after that, hold still (LOST) until this many missed frames, then search")
ap.add_argument("--bearing-age", type=float, default=2.5,
                help="s; drive only on a bearing measured on a settled frame at most this old")
ap.add_argument("--local-offsets", default="0,-20,20,-40,40",
                help="deg around the area where the target was last seen; searched first")
ap.add_argument("--local-rounds", type=int, default=2, help="repeat the local search this often before a full sweep")
ap.add_argument("--sweep", default="-90,-60,-30,0,30,60,90", help="full sweep, deg (run in snake order)")
ap.add_argument("--look-frames", type=int, default=2,
                help="at each search position, wait for this many sharp (settled) detector frames")
ap.add_argument("--sweep-dwell", type=float, default=4.0, help="s; max time at one search position")
ap.add_argument("--spin-time", type=float, default=1.2)
ap.add_argument("--spin-speed", type=float, default=0.05)
ap.add_argument("--no-drive", action="store_true")
ap.add_argument("--log", default="approach_log.csv")
ap.add_argument("--headless", action="store_true", help="no video window (SSH / tests); stop with Ctrl+C")
ap.add_argument("--duration", type=float, default=0.0, help="s; quit after this long (0 = run until stopped)")
args = ap.parse_args()

GLOBAL_SWEEP = [float(a) for a in args.sweep.split(",")]
LOCAL_OFFSETS = [float(a) for a in args.local_offsets.split(",")]

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

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.CONFLATE, 1)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(f"tcp://{args.robot}:5555")
push = ctx.socket(zmq.PUSH)
push.connect(f"tcp://{args.robot}:5556")

print("[1/3] loading model...", flush=True)
detector = TextDetector(args.prompt, model=args.model, noun=args.noun, attr_check=args.attr_check,
                        conf=args.attr_conf if args.attr_check else min(args.conf, args.lock_conf),
                        colour_check=not args.no_colour_check)
if detector.colour:
    print(f"first lock checks the colour '{detector.colour}' with CLIP", flush=True)
tracker = TemplateTracker(min_score=args.trk_score)
if detector.attr_check:
    print(f"detecting '{detector.noun}', CLIP checks the crops against '{args.prompt}'", flush=True)
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


def ground_dist(u, y_bottom, tilt_cmd, h, fx, fy, cx, cy):
    """Distance (m, along the floor) from the camera to the point where the box's bottom edge touches the floor.
    Assumes the object stands on the floor. If the bottom edge is cut off by the image border, the true base is
    even lower, so the value is an upper bound -- still safe for the stop decision. None near/above the horizon."""
    _, below_axis = pixel_to_angles(u, min(y_bottom, h - 1), fx, fy, cx, cy)   # deg, + = below the optical axis
    depression = -args.tilt_sign * tilt_cmd + below_axis                       # deg below the horizon
    if depression < 2.0:
        return None
    return args.cam_height / math.tan(math.radians(depression))


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
# all settings of this run next to the log (the prompt was missing from the logs of the 2026-10-24 runs)
with open(os.path.splitext(args.log)[0] + "_args.json", "w") as f:
    json.dump(dict(vars(args), started=time.strftime("%Y-%m-%d %H:%M:%S")), f, indent=1)
log = csv.writer(log_f)
log.writerow(["frame_id", "t_capture", "state", "det_ms", "found", "box_ratio", "dist_m",
              "pan_cmd", "tilt_cmd", "yaw_ff_deg", "L", "R", "moving", "det_mode", "n_dets", "src", "conf", "colour_p"])

state = "SEARCH"
last_seen = 0.0
last_pan_seen = 0.0          # bearing of the target relative to the body, from settled frames only
last_box = None              # previous target box (image coords), for association and the zoomed crop
bearing_t = 0.0              # when last_pan_seen was last measured on a settled frame
target_dist = None           # ground distance to the target's base, m (settled frames only), None = unknown
misses = 0                   # consecutive frames without the target
since_det = 0                # frames since the detector itself last found the target
confirm_box = None           # candidate being confirmed before the first lock
confirm_pass = confirm_n = 0 # frames it passed / frames spent confirming
recheck_n = recheck_fail = 0 # colour re-check while tracking
last_box_angles = (0.0, 0.0) # gimbal pan/tilt the frame of last_box was labelled with
PX_PER_DEG = None            # image shift per degree of gimbal motion near the centre (fx * pi / 180)
area_pan = None              # where (body-relative pan) the target was last seen, coarse; searched first
prev_state = state
search_list = []             # current search plan (pan angles); empty = make a new one
search_phase = "global"      # "local" (around area_pan) or "global" (full sweep)
local_round = 0
search_idx = 0
look_count = 0               # settled frames looked at in the current search position
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

t_start = time.time()
while not stop_requested:
    if args.duration and time.time() - t_start > args.duration:
        break
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
    recent = last_box is not None and misses <= args.lost_frames
    if PX_PER_DEG is None:
        PX_PER_DEG = fx * math.pi / 180.0
    # where the target should be now: the last box moved by the gimbal motion since then (a 35 deg centring move
    # shifts it ~170 px, more than the same-object test allows)
    pred_box, shift = last_box, (0.0, 0.0)
    if recent:
        shift = (-args.pan_sign * (hdr["pan_cmd"] - last_box_angles[0]) * PX_PER_DEG,
                 args.tilt_sign * (hdr["tilt_cmd"] - last_box_angles[1]) * PX_PER_DEG)
        pred_box = [last_box[0] + shift[0], last_box[1] + shift[1], last_box[2] + shift[0], last_box[3] + shift[1]]
    if (state in ("SEARCH", "CONFIRM") or misses > 0) and not args.no_tiles:
        # ~0.7-1.1 s per frame: while searching, and while the body is stopped waiting after a miss (some objects,
        # like the red pouf, are found almost only with the zoom); never while driving on a fresh detection
        mode = "tiles"
    elif recent and last_box[3] - last_box[1] < args.roi_px:
        mode = "roi"
    else:
        mode = "full"
    dets = detector(frame, mode=mode, last_box=pred_box if recent else None)
    def near(d, ref):
        return math.hypot((d[0] + d[2] - ref[0] - ref[2]) / 2, (d[1] + d[3] - ref[1] - ref[3]) / 2) < args.assoc_px

    def same_object(d, ref):
        """Overlaps the last box, or is close relative to its size AND of similar size. (run11: a box 5x smaller
        than the chair, 150 px away, was taken as 'near' -> the lock moved to a white box, then to the pouf.)
        Weak detections (score below --conf) must overlap."""
        o = iou(d, ref)
        if o >= 0.3:
            return True
        if d[5] < args.conf:
            return False
        rw, rh = ref[2] - ref[0], ref[3] - ref[1]
        hr = (d[3] - d[1]) / max(rh, 1.0)
        dist = math.hypot((d[0] + d[2] - ref[0] - ref[2]) / 2, (d[1] + d[3] - ref[1] - ref[3]) / 2)
        return 0.6 <= hr <= 1.7 and dist < 0.75 * max(rw, rh) + 60

    locked = recent and state in ("TRACK", "ARRIVED", "LOST")
    colour_p = None
    if locked:
        # relaxed once locked: any box next to the target counts, down to --lock-conf (the score flickers); boxes
        # elsewhere are ignored, so it never switches to another object while tracking (run10: jumped to the pouf)
        dets = [d for d in dets if same_object(d, pred_box) or same_object(d, last_box)]
        dets.sort(key=lambda d: -max(iou(d, pred_box), iou(d, last_box)))
    else:
        # strict first lock / re-acquisition: higher score, next to the candidate being confirmed, right colour
        dets = [d for d in dets if d[5] >= args.acquire_conf and (state != "CONFIRM" or near(d, confirm_box))]
        if dets and detector.colour:
            ps = detector.colour_prob(frame, dets)
            colour_p = max(ps)
            dets = [d for d, p in zip(dets, ps) if p >= args.colour_min]
    src = "det" if dets else ""
    close = recent and (last_box[3] - last_box[1]) >= args.close_ratio * frame.shape[0]
    if not dets and locked and since_det < (args.trk_frames_close if close else args.trk_frames):
        box, score = tracker.update(frame, around=pred_box)   # detector missed: last patch, where it should be
        if box is not None:
            dets, src = [box + [score, score, 0.0]], "trk"
    since_det = 0 if src == "det" else since_det + 1     # the tracker never extends its own budget
    lock_dropped = False
    if locked and src == "det" and detector.colour and args.recheck_every:
        # identity safety net: is the tracked object still the asked colour? (V4 will add a visual template)
        recheck_n += 1
        if recheck_n % args.recheck_every == 0:
            colour_p = detector.colour_prob(frame, dets[:1])[0]
            recheck_fail = recheck_fail + 1 if colour_p < args.recheck_min else 0
            if recheck_fail >= args.recheck_fails:
                print(f"colour re-check failed {recheck_fail}x (p={colour_p:.2f}) -> dropping the lock", flush=True)
                dets, src, lock_dropped = [], "", True
                recheck_fail = 0
    if not locked:
        recheck_n = recheck_fail = 0
    if not recent:
        tracker.reset()
    det_ms = (time.perf_counter() - t0) * 1000.0
    now = time.time()
    dt = now - t_prev
    t_prev = now

    # ---- yaw feed-forward: how much did the body rotate since the last update? ----
    yaw_ff = args.yaw_comp * chassis_yaw_rate_deg() * dt
    last_pan_seen -= yaw_ff      # the target's bearing relative to the body shrinks as we turn toward it
    if area_pan is not None:
        area_pan -= yaw_ff

    found = len(dets) > 0
    misses = 0 if found else misses + 1
    if lock_dropped:
        misses = args.lost_frames + 1            # straight back to the strict search, no grace / LOST hold
        last_box = None
    prev_state = state
    box_ratio = None
    L = R = 0.0

    # first lock: the candidate has to pass the strict checks in --confirm-frames of --confirm-window frames
    if found and not locked:
        if state != "CONFIRM":
            confirm_pass = confirm_n = 0
        confirm_pass += 1
        confirm_box = dets[0][:4]
    if state == "CONFIRM" or (found and not locked):
        confirm_n += 1
    confirmed = found and not locked and confirm_pass >= args.confirm_frames
    if confirmed:
        last_box, recent = confirm_box, True     # lock onto exactly the confirmed object

    if hold:
        state = "HOLD"
        L, R = send_drive(0.0, 0.0)

    elif found and not locked and not confirmed:
        state = "CONFIRM"                        # candidate: stand still and look again
        L, R = send_drive(0.0, 0.0)
        x1, y1, x2, y2 = confirm_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)

    elif found:
        last_seen = now
        best = dets[0]                     # highest score
        if recent:
            # stay on the same object: nearest box to the previous one (two similar objects -> no switching)
            pcx, pcy = (last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2
            box_dist = lambda d: math.hypot((d[0] + d[2]) / 2 - pcx, (d[1] + d[3]) / 2 - pcy)
            near = min(dets, key=box_dist)
            if box_dist(near) < args.assoc_px:
                best = near
        x1, y1, x2, y2 = best[:4]
        last_box = best[:4]
        last_box_angles = (hdr["pan_cmd"], hdr["tilt_cmd"])
        if src == "det":
            tracker.init(frame, best[:4])      # refreshed by real detections only (no drift onto background)
        u, v = (x1 + x2) / 2, (y1 + y2) / 2
        box_ratio = (y2 - y1) / h
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0) if src == "det" else (255, 200, 0), 2)

        err_pan, err_tilt = pixel_to_angles(u, v, fx, fy, cx, cy)

        # bearing of the target relative to the body at capture time, then corrected for rotation since
        target_pan = hdr["pan_cmd"] + args.pan_sign * err_pan - yaw_ff
        target_tilt = hdr["tilt_cmd"] - args.tilt_sign * err_tilt
        if not moving:
            # only a settled frame shows the view at hdr["pan_cmd"]; in-flight frames would count the error twice
            last_pan_seen = target_pan
            area_pan = target_pan
            bearing_t = now
            # the tilt in the header is the real one only on settled frames, same as the pan
            target_dist = ground_dist((x1 + x2) / 2, y2, hdr["tilt_cmd"], h, fx, fy, cx, cy)

        elif prev_state in ("SEARCH", "CONFIRM"):
            # found in a frame taken while the camera swept: the exact angle is unknown, but the camera is on its way
            # to gimbal_pan -- remember that area and stop the sweep there (no correction until a settled frame)
            area_pan = gimbal_pan

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

        if (target_dist is not None and target_dist <= args.arrive_dist) or box_ratio >= args.arrive_ratio:
            state = "ARRIVED"
            L, R = send_drive(0.0, 0.0)
        else:
            state = "TRACK"
            bearing = last_pan_seen
            if now - bearing_t > args.bearing_age:
                # just locked from a frame taken while the gimbal moved: no trustworthy bearing yet, don't drive
                L, R = send_drive(0.0, 0.0)
            else:
                turn = 0.0 if abs(bearing) < args.turn_deadband else args.k_turn * bearing
                if abs(bearing) < args.align_deg:
                    if target_dist is not None:
                        # full speed until --slow-dist before the stop point, then down to 25%
                        fwd = args.max_speed * clamp((target_dist - args.arrive_dist) / args.slow_dist, 0.25, 1.0)
                    else:
                        fwd = args.max_speed * max(0.3, 1.0 - box_ratio / args.arrive_ratio)
                else:
                    fwd = 0.0
                L, R = send_drive(fwd, turn)

    elif state == "CONFIRM" and confirm_n < args.confirm_window:
        L, R = send_drive(0.0, 0.0)              # candidate not seen in this frame: keep looking, standing still

    elif misses <= args.miss_frames and state in ("TRACK", "ARRIVED"):
        # a few missed frames: wait for the target with the body STOPPED and the camera held. (Repeating the last
        # command meant turning blind for up to ~3 s at 1 frame/s -> over-turned and lost the target up close.)
        L, R = send_drive(0.0, 0.0)
        if yaw_ff != 0.0:
            send_gimbal(hdr, gimbal_pan - yaw_ff, gimbal_tilt, limit_step=False, correction=False)

    elif misses <= args.lost_frames and state in ("TRACK", "ARRIVED", "LOST"):
        # lost: stop the body and hold the camera where it is (it already points at the last settled bearing
        # or is moving there); sending it to a bearing from an in-flight frame is what made it jump before
        state = "LOST"
        L, R = send_drive(0.0, 0.0)

    else:
        state = "SEARCH"
        target_dist = None
        if prev_state not in ("SEARCH", "CONFIRM"):    # a rejected candidate: just continue the sweep
            # (re)start: around the area where the target was seen, if we have seen it at all
            search_list, local_round = [], 0
            search_phase = "local" if area_pan is not None else "global"

        if now < spin_until:
            send_gimbal(hdr, 0.0, 0.0)
            L, R = send_drive(0.0, args.spin_speed)
        else:
            if search_list and (look_count >= args.look_frames or now - search_t0 > args.sweep_dwell):
                search_idx += 1          # looked long enough here -> next position
                look_count, search_t0 = 0, now
                if search_idx >= len(search_list):
                    search_list = []
                    if search_phase == "local" and local_round + 1 < args.local_rounds:
                        local_round += 1
                    elif search_phase == "local":
                        search_phase = "global"
                    else:
                        # full sweep failed: turn the body a bit (if driving is allowed), then look around the
                        # remembered area again (it moves with the body via yaw feed-forward) or sweep again
                        if not args.no_drive:
                            spin_until = now + args.spin_time
                        search_phase, local_round = ("local", 0) if area_pan is not None else ("global", 0)
            if not search_list:
                if search_phase == "local":
                    search_list = [clamp(area_pan + o, -180.0, 180.0) for o in LOCAL_OFFSETS]
                else:
                    # snake order: start from the end nearer to where the camera points (no 180 deg jump back)
                    search_list = list(GLOBAL_SWEEP)
                    if abs(gimbal_pan - search_list[-1]) < abs(gimbal_pan - search_list[0]):
                        search_list.reverse()
                search_idx, look_count, search_t0 = 0, 0, now
            if now >= spin_until:
                send_gimbal(hdr, search_list[search_idx], 0.0, limit_step=False)
                if not moving and pending is None:
                    look_count += 1      # this frame was sharp and already taken at this position
            L, R = send_drive(0.0, 0.0)

    log.writerow([hdr["frame_id"], f"{hdr['t_capture']:.6f}", state, f"{det_ms:.1f}", int(found),
                  box_ratio, None if target_dist is None else f"{target_dist:.2f}", f"{gimbal_pan:.2f}", f"{gimbal_tilt:.2f}", f"{yaw_ff:.2f}", f"{L:.3f}", f"{R:.3f}", int(moving), mode, len(dets), src,
                  f"{dets[0][5]:.2f}" if dets else None, None if colour_p is None else f"{colour_p:.2f}"])

    cv2.drawMarker(frame, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    color = {"SEARCH": (0, 200, 255), "TRACK": (0, 255, 0), "LOST": (0, 0, 255), "ARRIVED": (255, 0, 255),
             "CONFIRM": (0, 165, 255),
             "HOLD": (200, 200, 200)}[state]
    txt = f"{state}{' PAUSED' if paused else ''}{' moving' if moving else ''}  det {det_ms:.0f}ms  pan {gimbal_pan:.0f}  L {L:.2f} R {R:.2f}"
    if box_ratio is not None:
        txt += f"  box {box_ratio:.2f}/{args.arrive_ratio:.2f}"
    if target_dist is not None:
        txt += f"  dist {target_dist:.2f}/{args.arrive_dist:.2f} m"
    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if args.headless:
        continue
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
if not args.headless:
    cv2.destroyAllWindows()
