# eval/ -- quick ground truth, baseline and detector numbers (midterm, 2026-10-24)

Everything here runs offline on the recordings (`rec/`, 29 sequences of the dataset; `S1_seq01` is an old test
take and is excluded). Timings: laptop i7-12650H, CPU only.

```
python eval/candidates.py --rec rec --step 30   # candidate boxes + contact sheets (eval/sheets/)
# a person writes eval/gt_raw.txt from the sheets (format in make_gt.py)
python eval/make_gt.py                          # -> eval/gt.json
python eval/run_detectors.py                    # -> eval/dets/<config>.json
python eval/metrics.py                          # -> tables below, eval/results.json
python eval/size_study.py --every 2             # -> object-size table below, eval/size_results.json
```

## Ground truth (quick, not CVAT)

One frame every 2 s (every 30th frame) of every sequence. Candidate boxes come from YOLO-World s at 640 and 1280
(conf 0.05); one person picked the target box per frame, or marked it "visible, no fitting box" (mostly motion
blur) or "not in frame". 298 frames of the dataset: 232 with a box, 41 visible without box, 25 target absent.
Target = the dark box; in S5 the second (grey, see-through) box is the distractor.
Limits: one annotator; boxes are detector boxes, so configurations similar to s640/s1280 have a small advantage;
frames where no detector found the target have no box and are excluded from detector scores.

## V1 baseline, closed loop (the recordings were made while V1 drove the gimbal)

| scenario | frames | target in frame | centring error mean / p95 | near centre (<10 deg) |
|---|---|---|---|---|
| S1 slow | 50 | 100% | 10.5 / 34.2 deg | 69% |
| S2 fast | 50 | 96% | 17.7 / 39.4 deg | 33% |
| S3 occlusion | 50 | 90% | 10.0 / 50.5 deg | 74% |
| S4 leaves frame | 51 | 71% | 20.3 / 54.1 deg | 38% |
| S5 distractor | 50 | 94% | 14.0 / 37.6 deg | 43% |
| S6 low light | 47 | 100% | 10.9 / 41.8 deg | 66% |
| **all** | 298 | 92% | 13.3 / 46.0 deg | 56% |

Centring error = angle of the target's box centre from the optical axis, through the calibration (undistorted).
The proposal's goal for the full system is >= 90% in frame in S1-S4; V1 already has it except S4 (71%), but the
centring is poor when the target moves (S2) or leaves (S4).

## Detector configurations, open loop on the same frames

| config | ms/frame median (p95) | correct lock | wrong lock | missed | false alarm (target absent) | correct lock S5 / S6 |
|---|---|---|---|---|---|---|
| s640 (= V1) | 102 (107) | 78% | 15% | 8% | 76% | 62% / 63% |
| s1280 | 425 (435) | 59% | 26% | 16% | 80% | 41% / 61% |
| s640 + tiles | 693 (717) | 78% | 18% | 4% | 88% | 54% / 79% |
| m640 | 265 (277) | 74% | 9% | 17% | 56% | 49% / 63% |
| l640 | 551 (557) | 50% | 3% | 47% | 36% | 19% / 34% |
| x640 | 806 (839) | 57% | 4% | 39% | 48% | 30% / 29% |
| s640 + attribute check | 275* (513) | 5% | 13% | 82% | 44% | 3% / 8% |

Correct lock = the most confident box has IoU >= 0.5 with the target; wrong lock = it is on something else while
the target is visible; false alarm = any box while the target is not in the frame. Threshold 0.10 (as on the
robot). *timed while another job was running.

What it says:
- **YOLO-World s at 640 stays the main detector**: the fastest and the best correct-lock rate. Bigger models are
  2.5-8x slower on this CPU and, at the same threshold, miss more (they are more conservative); upscaling to
  1280 finds more boxes but locks on the wrong one more often.
- **The detector almost always "sees" something**: 76% of frames without the target still get a "dark box"
  (monitors, black devices). This is what swings the camera away in S4 and the main argument for V4.
- **Attribute check fails for this prompt**: CLIP is asked whether the crop fits "a dark box" better than "a box".
  It separates a yellow chair from red/beige chairs, but here the distractors are dark too and the real box is
  removed (82% missed). Not general, so `--attr-check` is off by default.

## Object size (= distance) study

The frames with a verified box (every 2nd), shrunk by 1x, 2x, 3x, 4x and padded back to 640x480 (the same scene
2-4x farther away). Correct lock by the target's box height after shrinking; any object class works this way.
With our lens (fx ~ 273 px) a 0.4 m object is ~22 px tall at 5 m and ~11 px at 10 m.

SIZE_TABLE

- Below ~40 px the plain full-frame detector (V1) mostly fails -- this is the "can't find it beyond ~5 m" problem.
- Zoomed tiles (used in SEARCH, ~7x the time) and a zoomed crop around the last box (used in TRACK for small
  targets, ~2.5x) recover a large part. ROI uses the previous box (here from GT; on the robot the last detection).
- Upscaling the whole frame to 1280 is not the answer: slower than ROI and worse on medium-size objects.
