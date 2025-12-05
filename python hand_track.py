
import cv2
import numpy as np
import time
import math
from collections import deque

# ---------------------------
# Parameters / Config (tuneable)
# ---------------------------
CAM_W, CAM_H = 640, 480
proc_w, proc_h = 320, 240           # processing resolution (smaller = faster)
BG_CAPTURE_FRAMES = 5
MOTION_THRESH = 30
MIN_CONTOUR_AREA = 1200             # ignore tiny blobs
SKIP_SKIN_EVERY_N = 2               # skip skin seg every N frames when fps low
DANGER_DIST_PIX = 30.0              # in display pixels (tight)
WARNING_DIST_PIX = 80.0             # in display pixels (approaching)
# Note: thresholds refer to display-space pixels (640x480)

# Virtual object (display coords)
obj_x, obj_y, obj_w, obj_h = 400, 150, 100, 100
obj_cx, obj_cy = obj_x + obj_w // 2, obj_y + obj_h // 2

# Smoothing / stability
CENTROID_SMOOTH_ALPHA = 0.5         # exponential smoothing for centroid
DIST_WINDOW = 6                     # moving average window for distance
STATE_DEBOUNCE_FRAMES = 6          # require N consistent frames to change state

# Misc
FRAME_WAIT = 1                      # cv2.waitKey delay
SHOW_MASK_WINDOW = False            # set True to debug masks (costly)
LOW_FPS_THRESHOLD = 8.0

# ---------------------------
# Helpers
# ---------------------------
def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_skin_mask(hsv):
    # tuned HSV ranges for typical skin tones; still empirical
    lower1 = np.array([0, 48, 30], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 48, 30], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    skin = cv2.bitwise_or(m1, m2)
    # cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, k, iterations=1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k, iterations=1)
    return skin

def rect_min_distance_to_points(rx, ry, rw, rh, points):
    """
    Compute minimal Euclidean distance between any point and the rectangle boundary.
    If any point inside rectangle -> distance = 0.
    Points are (x,y) in same coords as rectangle.
    """
    if points is None or len(points) == 0:
        return float('inf')
    px = points[:,0]
    py = points[:,1]
    # Check inside
    inside_mask = (px >= rx) & (px <= rx+rw) & (py >= ry) & (py <= ry+rh)
    if np.any(inside_mask):
        return 0.0
    # For points outside, compute distance to rectangle (shortest distance)
    # For point (x,y), dx = max(rx-x, 0, x-(rx+rw)), similarly for y
    dx = np.maximum(np.maximum(rx - px, 0), px - (rx + rw))
    dy = np.maximum(np.maximum(ry - py, 0), py - (ry + rh))
    d = np.hypot(dx, dy)
    return float(np.min(d))

# ---------------------------
# Init camera
# ---------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
time.sleep(0.3)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# ---------------------------
# Capture background median (proc res)
# ---------------------------
bg_frames = []
print(f"Capturing {BG_CAPTURE_FRAMES} background frames — keep scene static...")
captured = 0
while captured < BG_CAPTURE_FRAMES:
    ret, frame = cap.read()
    if not ret:
        continue
    small = cv2.resize(frame, (proc_w, proc_h))
    bg_frames.append(small)
    captured += 1
    cv2.putText(frame, f"Capturing background {captured}/{BG_CAPTURE_FRAMES}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Background capture", frame)
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break
cv2.destroyWindow("Background capture")
bg_median = np.median(np.stack(bg_frames, axis=3), axis=3).astype(np.uint8)

# ---------------------------
# Main loop state
# ---------------------------
prev_time = time.time()
fps_smooth = 0.0
frame_count = 0

smoothed_centroid = None  # in display coords
dist_buffer = deque(maxlen=DIST_WINDOW)

# State machine with debounce counters
state = "SAFE"
state_color = (0,255,0)
state_candidate = state
state_candidate_count = 0

print("Starting main loop. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # FPS calc + smoothing
    now = time.time()
    dt = now - prev_time if prev_time else 1e-6
    fps = 1.0 / dt if dt > 0 else 0.0
    prev_time = now
    fps_smooth = fps if fps_smooth == 0.0 else (0.92 * fps_smooth + 0.08 * fps)

    # Resize for processing
    small = cv2.resize(frame, (proc_w, proc_h))

    # Motion mask
    diff = cv2.absdiff(small, bg_median)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, motion_mask = cv2.threshold(gray, MOTION_THRESH, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)
    motion_mask = cv2.medianBlur(motion_mask, 5)

    # Skin mask (skip sometimes when FPS low)
    do_skin = True
    if fps_smooth < LOW_FPS_THRESHOLD:
        do_skin = (frame_count % SKIP_SKIN_EVERY_N) == 0

    if do_skin:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        skin_mask = get_skin_mask(hsv)
    else:
        skin_mask = motion_mask.copy()

    # Combine
    combined = cv2.bitwise_and(motion_mask, skin_mask)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    # Contours on processing resolution
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    if contours:
        # choose largest contour by area
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            # convex hull to stabilize fingertip region
            hull = cv2.convexHull(c)
            best_cnt = hull.reshape(-1, 2)  # Nx2 array (proc coords)

    # Scale best contour points to display coords for distance measurement and drawing
    scale_x = CAM_W / proc_w
    scale_y = CAM_H / proc_h
    display_contour_pts = None
    if best_cnt is not None:
        pts = best_cnt.astype(np.float32)
        pts[:,0] = pts[:,0] * scale_x
        pts[:,1] = pts[:,1] * scale_y
        display_contour_pts = pts.astype(np.int32)

        # centroid from moments on the original contour (not hull) if available
        # falling back to mean of hull points for center
        cx_disp = int(np.mean(display_contour_pts[:,0]))
        cy_disp = int(np.mean(display_contour_pts[:,1]))

        # smooth centroid
        if smoothed_centroid is None:
            smoothed_centroid = (cx_disp, cy_disp)
        else:
            sx = int(CENTROID_SMOOTH_ALPHA * smoothed_centroid[0] + (1 - CENTROID_SMOOTH_ALPHA) * cx_disp)
            sy = int(CENTROID_SMOOTH_ALPHA * smoothed_centroid[1] + (1 - CENTROID_SMOOTH_ALPHA) * cy_disp)
            smoothed_centroid = (sx, sy)
    else:
        # no hand detected this frame -> slowly decay smoothed_centroid to None
        if smoothed_centroid is not None:
            # keep last known for a few frames (optional), here we keep it
            pass

    # Compute distance between hand and virtual object's rectangle (display coords)
    if display_contour_pts is not None:
        d = rect_min_distance_to_points(obj_x, obj_y, obj_w, obj_h, display_contour_pts)
    else:
        d = float('inf')

    # Feed distance buffer and compute moving average for stability
    if math.isfinite(d):
        dist_buffer.append(d)
    else:
        # fill with large values when no detection so state goes to SAFE gradually
        dist_buffer.append(9999.0)

    avg_d = float(np.mean(dist_buffer)) if len(dist_buffer) > 0 else float('inf')

    # Determine candidate state by thresholds (use avg_d)
    if avg_d <= DANGER_DIST_PIX:
        candidate = "DANGER"
    elif avg_d <= WARNING_DIST_PIX:
        candidate = "WARNING"
    else:
        candidate = "SAFE"

    # Debounce: require STATE_DEBOUNCE_FRAMES consistent frames to switch
    if candidate == state:
        state_candidate = state
        state_candidate_count = 0
    else:
        if candidate == state_candidate:
            state_candidate_count += 1
        else:
            state_candidate = candidate
            state_candidate_count = 1

        if state_candidate_count >= STATE_DEBOUNCE_FRAMES:
            state = state_candidate
            state_candidate_count = 0

    # Map to colors
    if state == "SAFE":
        state_color = (0,255,0)
    elif state == "WARNING":
        state_color = (0,200,200)
    else:
        state_color = (0,0,255)

    # ---------------------------
    # DRAWING (display frame)
    # ---------------------------
    display = frame.copy()

    # Virtual object
    cv2.rectangle(display, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), (0,255,0), 2)
    cv2.line(display, (obj_x, obj_y), (obj_x+obj_w, obj_y+obj_h), (0,255,0), 1)  # small cross for visibility

    # Draw hand contour (if any)
    if display_contour_pts is not None:
        cv2.polylines(display, [display_contour_pts], True, (0,120,255), 2)
        # draw smoothed centroid
        if smoothed_centroid is not None:
            cv2.circle(display, smoothed_centroid, 6, (0,0,255), -1)

    # Draw line from centroid to object center in WARNING/DANGER
    if smoothed_centroid is not None and state in ("WARNING", "DANGER"):
        cv2.line(display, smoothed_centroid, (obj_cx, obj_cy), (255,0,0), 2)

    # Distance text
    if math.isfinite(avg_d) and avg_d < 9999:
        cv2.putText(display, f"d={int(avg_d)}px", (10, CAM_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # State overlay (top-left)
    if state == "SAFE":
        cv2.putText(display, "SAFE", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
    elif state == "WARNING":
        cv2.putText(display, "WARNING", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3)
    else:
        # flashing danger text (toggle every few frames)
        flash_period = 10
        visible = (frame_count % flash_period) < (flash_period // 2)
        if visible:
            cv2.putText(display, "DANGER DANGER", (30, CAM_H//2 - 20), cv2.FONT_HERSHEY_DUPLEX, 1.8, state_color, 5)
        else:
            cv2.rectangle(display, (5,5), (CAM_W-5, CAM_H-5), state_color, 2)

    # FPS & hints
    cv2.putText(display, f"FPS: {fps_smooth:.1f}", (CAM_W - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if fps_smooth < LOW_FPS_THRESHOLD:
        cv2.putText(display, "LOW FPS - skipping skin frames", (10, CAM_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Show windows
    if SHOW_MASK_WINDOW:
        mask_vis = cv2.resize(combined, (320, 240))
        cv2.imshow("Mask (combined)", mask_vis)

    cv2.imshow("Hand Tracking (stable)", display)

    # key events
    key = cv2.waitKey(FRAME_WAIT) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):  # toggle mask window for debugging
        SHOW_MASK_WINDOW = not SHOW_MASK_WINDOW

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Exited.")
