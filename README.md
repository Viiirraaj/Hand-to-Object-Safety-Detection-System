# Hand-to-Object Safety Detection System

A lightweight real-time computer vision system that detects how close your hand is to a virtual object using a laptop webcam. Displays three states based on distance:

* **SAFE** – Hand is far from the object
* **WARNING** – Hand is approaching
* **DANGER** – Hand extremely close or touching the virtual boundary

This system is optimized for low-spec laptops and automatically adjusts processing for smoother FPS.

---

## Features

* Real-time hand tracking using webcam
* Motion + skin detection hybrid pipeline
* Auto optimization when FPS drops
* Convex hull hand detection
* Distance measurement to virtual object
* Debounced SAFE/WARNING/DANGER state switching
* Clean, short, and optimized code

---

## Installation

### 1. Install Dependencies

```
pip install opencv-python numpy
```

### 2. Run the Program

```
python hand_safety.py
```

Press **Q** to exit the program.

---

## How It Works

### 1. Background Median Capture

Captures a few initial frames to create a background model.

### 2. Motion Detection

Frame differencing + thresholding extracts moving regions.

### 3. Skin Detection (HSV-based)

Used when FPS is stable; automatically skipped on low FPS to prevent lag.

### 4. Convex Hull Extraction

Largest moving skin region is identified as the hand.

### 5. Distance Computation

Calculates the distance between the hand contour and a virtual rectangle.

### 6. State Machine

The system outputs:

* SAFE
* WARNING
* DANGER

Using rolling average, distance thresholds, and debounce stability logic.

---

## Key Configuration Parameters

| Parameter         | Description                          |
| ----------------- | ------------------------------------ |
| `DANGER_D = 30`   | Distance threshold for DANGER        |
| `WARNING_D = 80`  | Distance threshold for WARNING       |
| `proc_w, proc_h`  | Preprocessing resolution             |
| `CENTROID_ALPHA`  | Exponential smoothing for centroid   |
| `DIST_WINDOW = 6` | Rolling average for distance         |
| `DEBOUNCE = 6`    | Frames needed to confirm a new state |

---

## System Behavior Optimization

The code adapts depending on FPS:

| FPS       | Behavior                         |
| --------- | -------------------------------- |
| `< 8 FPS` | Skip skin detection to avoid lag |
| Normal    | Use full motion+skin pipeline    |

Designed to run smoothly even on basic hardware.

---

## Project Structure

```
.
├── python hand_track.py      # Main code
└── README.md           # Documentation
```

---

## Technologies Used

* Python
* OpenCV
* NumPy

---

## Future Improvements

* Multi-hand detection
* Gesture classification
* 3D depth estimation
* AR/VR integration
* Dynamic on-screen object placement

---

## Author

**Viraj Shinde**
Master’s Student (Computer Science)
AI/ML & Computer Vision Projects


