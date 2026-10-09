"""
calibrate.py  -- runs ON THE LAPTOP, uses the robot stream (same resolution/JPEG as the experiments).

1) Print a checkerboard (default: 9x6 INNER corners, 25 mm squares), tape it flat to cardboard.
2) python calibrate.py --robot 192.168.4.1 --cols 9 --rows 6 --square 0.025
3) Show the board at different distances, angles and image corners. SPACE = save view (green = corners found).
   Collect 20-30 views, then press q. Writes calib.npz (K, dist) and prints RMS reprojection error (aim < 0.5 px).
"""
import argparse

import cv2
import numpy as np
import zmq

ap = argparse.ArgumentParser()
ap.add_argument("--robot", required=True)
ap.add_argument("--cols", type=int, default=9)
ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square", type=float, default=0.025)
ap.add_argument("--out", default="calib.npz")
args = ap.parse_args()

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.CONFLATE, 1)
sub.setsockopt(zmq.SUBSCRIBE, b"")
sub.connect(f"tcp://{args.robot}:5555")

pattern = (args.cols, args.rows)
objp = np.zeros((args.cols * args.rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
obj_pts, img_pts = [], []
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
size = None

while True:
    _, jpg = sub.recv().split(b"\n", 1)
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ok, corners = cv2.findChessboardCorners(gray, pattern, None)
    vis = frame.copy()
    if ok:
        cv2.drawChessboardCorners(vis, pattern, corners, ok)
    cv2.putText(vis, f"saved: {len(obj_pts)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("calibration", vis)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" ") and ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
    if k == ord("q"):
        break

cv2.destroyAllWindows()
rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
np.savez(args.out, K=K, dist=dist, size=np.array(size))
print("RMS reprojection error [px]:", rms)
print("K =\n", K)
print("dist =", dist.ravel())
fx = K[0, 0]
print(f"horizontal FOV ~ {2 * np.degrees(np.arctan(size[0] / (2 * fx))):.1f} deg")
