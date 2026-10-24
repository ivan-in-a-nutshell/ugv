# Text-guided search and approach for a UGV

A Waveshare UGV Beast is given a text description ("a yellow chair"), searches for the object with its
pan-tilt camera, locks on it and drives up to it. The detector is open-vocabulary (YOLO-World), so the target is
changed by typing a sentence, without training data. The core problem is that this detector is slow on our
hardware (~100 ms per frame on the laptop CPU, delayed by Wi-Fi), while the gimbal and the chassis need a smooth
signal. The project builds fast components around it and measures what each one adds (V1-V4).

Team: Arsen Botsko, Ivan Zarytskyi, Zakhar Veresniuk (UCU, AI course 2026). Proposal:
`Proposal_Veresniuk_Botsko_Zarytskyi-1.pdf`. Operational details (networks, setup, tuning): `HANDOVER.md`.

## Pipeline

```mermaid
flowchart LR
  subgraph Robot["Robot: Raspberry Pi + ESP32"]
    CAM[USB camera 640x480, 15 fps] --> BR[robot_bridge.py<br/>timestamp at capture, JPEG]
    BR -- UART JSON T:133 / T:1 --> ESP[ESP32<br/>gimbal + tracks]
  end
  subgraph Laptop
    DET[detector.py<br/>YOLO-World, text prompt<br/>+ zoom tiles / ROI<br/>for far objects] --> CTRL[laptop_approach.py<br/>settle gating, search,<br/>gimbal + chassis control]
  end
  BR -- ZeroMQ tcp 5555<br/>frame + header pan/tilt --> DET
  CTRL -- ZeroMQ tcp 5556<br/>pan/tilt, L/R --> BR
```

State machine in `laptop_approach.py`: `SEARCH` (gimbal sweep, zoomed tiles, then chassis spin) -> `TRACK`
(gimbal centres the target, chassis turns toward the bearing and drives) -> `ARRIVED` (box height / image
height >= `--arrive-ratio`). Short detection gaps are ignored (`--miss-grace`), longer ones go to `LOST`
(hold), then a local search around the last bearing, then the full sweep.

## Variants (study design from the proposal)

| Variant | What | Status |
|---|---|---|
| V1 | gimbal driven directly by the latest detection (baseline) | done, measured below |
| V1 + approach | search + drive to the object, settle gating, zoom for far objects | done, needs robot runs |
| V2 | + CSRT tracker between detections + PID, detector in its own thread | after midterm |
| V3 | + Kalman filter with delay compensation (late detections applied at capture time) | after midterm |
| V4 | + CLIP identity check on re-acquisition (template + text) | after midterm; text-only CLIP check tried (`--attr-check`), not general |

## Results so far (midterm)

Data: 30 recorded sequences, 6 scenarios x 5 (S1 slow, S2 fast, S3 occlusion, S4 target leaves the frame,
S5 distractor of the same class, S6 low light), 2 rooms, 20 s each at 15 fps, recorded while V1 drove the gimbal.
Quick ground truth: one frame every 2 s (308 frames), the target box chosen by hand among detector candidates
(`eval/candidates.py`, `eval/make_gt.py`). Full CVAT annotation is planned.

RESULTS_PLACEHOLDER

## Repository

| File | Runs on | Purpose |
|---|---|---|
| `robot_bridge.py` | Pi | camera stream, command forwarding, latency log, track speed cap + stop watchdog |
| `laptop_v1.py` | laptop | V1 baseline (gimbal only) |
| `record.py`, `run_seq.sh` | laptop | record every frame (+ V1 tracking) for the S1-S6 sequences |
| `laptop_approach.py` | laptop | search + gimbal + chassis approach (`--headless --duration` for tests) |
| `detector.py` | laptop | YOLO-World text detector + zoom tiles / ROI for far objects, optional CLIP attribute check |
| `replay_bridge.py` | laptop | fake bridge that streams a recording, to test the laptop scripts without the robot |
| `calibrate.py` | laptop | checkerboard calibration -> `calib.npz` (fx ~ 273 px, HFOV ~ 100 deg) |
| `eval/` | laptop | ground truth, detector benchmark, metrics (`eval/README.md`) |

## Quick start

```
pip install -r requirements.txt
# robot (ssh ws@192.168.50.5):  pkill -f app.py; ~/ugv_rpi/ugv-env/bin/python ~/robot_bridge.py
python laptop_approach.py --robot 192.168.50.5 --prompt "a yellow chair" --no-drive   # gimbal only first
python laptop_approach.py --robot 192.168.50.5 --prompt "a yellow chair"
# without the robot:
python replay_bridge.py --seq rec/S1_seq04 --loop &
python laptop_approach.py --robot 127.0.0.1 --prompt "a yellow chair" --no-drive --headless --duration 20
# metrics
python eval/run_detectors.py && python eval/metrics.py
```

Recordings (400 MB) are not in git; download them from the team Google Drive and link or unpack them as `rec/`.
