# ML-Based Human Attendance System

A high-throughput, edge-accelerated automated attendance system using RTSP IP video cameras. Faces are detected, tracked across frames, aligned, and identified in real time on an NVIDIA GPU; a trajectory state machine turns tripwire crossings into precise check-in and check-out records.

```text
Camera Stream (RTSP / Video)
   │
   ▼
YOLOv8-face (ONNX CUDA)  ──►  ByteTrack (Kalman Filter)  ──►  DFA Aligner (112×112 Canonical Warp)
   [~20 ms / frame]                [~0.5 ms / frame]                  [~7 ms / face]
                                                                            │
                                                                            ▼
State Machine & Attendance  ◄──  Virtual Tripwire  ◄──  AdaFace IR-50 (512-D Embedding)
   [Check-In / Check-Out]           [ENTER / EXIT]                    [~14 ms / face]
```

---

## Performance & Biometrics (NVIDIA RTX 5050 Laptop GPU)

| Metric | Measured Value | Standard / Target |
|---|---|---|
| **Pipeline Latency ($p_{50}$)** | **41.2 ms / frame** | $< 50 \text{ ms}$ (Real-time 20 FPS budget) |
| **Sustained Throughput** | **21.4 FPS continuous** | Easily keeps up with 15–20 FPS RTSP cameras |
| **Gallery Separation ($d'$)** | **11.9965** | $> 5.0$ is considered exceptional biometrics |
| **Rank-1 Identification** | **100.00%** | Top-1 leave-one-out recognition |
| **Weakest Genuine Match** | **0.6469** | Safe margin above false-reject floor |
| **Highest Impostor Match** | **0.2430** | Zero false accepts across enrolled gallery |
| **Active VRAM Footprint** | **~650 MB** | Lightweight; leaves over 5 GB free VRAM |

---

## Key Features

* **100% GPU Acceleration via ONNX Runtime**: Face detector (`yolov8n-face.onnx`), aligner (`dfa_mobilenet_aligner.onnx`), and recognizer (`adaface_ir50_base.onnx`) run entirely on the GPU via `CUDAExecutionProvider`. Fully compatible with NVIDIA Blackwell (`sm_120`), Ada Lovelace (`sm_89`), and Ampere (`sm_86`).
* **Multi-Frame Tracklet Consensus**: Does not rely on a single lucky frame. Associates detections across time into tracklets and requires $K$-of-$N$ agreeing recognition votes before committing an identity.
* **Trajectory Tripwire Crossing**: Evaluates movement direction vectors ($dx/dt$, depth growth) across a virtual trigger line to distinguish true entrances/exits from casual loitering.
* **Continuous Gallery Enrichment (`/gallery/review`)**: Human-in-the-loop review station that allows admins to approve sharp real-world corridor captures, with individualized mathematical safety thresholds per face.
* **Full English Web Dashboard**: Fast, responsive Bootstrap UI for employee management (add, inspect, delete), live camera viewing, daily logs, and CSV exports.

---

## Quick Start

### 1. Prerequisites
* Python 3.11
* NVIDIA GPU with CUDA 12+ and cuDNN 9+
* Windows 11 / Windows Server or Linux (Ubuntu 22.04 / 24.04)

### 2. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/lalitmahajn/ML-based-Human-attendance-system.git
cd ML-based-Human-attendance-system

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Linux: source venv/bin/activate

# Install dependencies (ensure onnxruntime-gpu is used, not cpu onnxruntime)
pip install -r requirements.txt
pip install onnxruntime-gpu==1.23.2
```

### 3. Download & Convert Models (`models/`)

The attendance pipeline requires 3 neural models in the `models/` folder. Because raw weights are excluded from Git, you can set them up automatically or manually:

#### Option A: One-Click Automated Setup (Recommended)
Run the built-in downloader and ONNX converter:
```powershell
python scripts/download_open_models.py
```
This automatically:
1. Downloads `yolov8n-face.pt` from HuggingFace.
2. Converts & exports `yolov8n-face.onnx` with dynamic axes (`batch, 3, height, width`) for GPU acceleration.
3. Downloads the open-source AdaFace IR-50 base recognizer (`adaface_ir50_base.onnx`).
4. Builds the DFA MobileNet canonical landmark aligner (`dfa_mobilenet_aligner.onnx`).
5. Validates tensor shapes and ONNX runtime input/output compatibility.

#### Option B: Manual Download & Conversion

If you prefer to download and convert the models manually:

1. **Face Detector (`yolov8n-face.onnx`)**:
   - Download PyTorch weights:
     ```powershell
     curl -L -o models/yolov8n-face.pt https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt
     ```
   - Convert to ONNX with dynamic input shapes:
     ```powershell
     # Using Ultralytics CLI:
     yolo export model=models/yolov8n-face.pt format=onnx imgsz=640 dynamic=True

     # Or using Python:
     python -c "from ultralytics import YOLO; YOLO('models/yolov8n-face.pt').export(format='onnx', imgsz=640, dynamic=True)"
     ```
     Ensure the exported file is placed at `models/yolov8n-face.onnx`.

2. **Face Recognizer (`adaface_ir50_base.onnx`)**:
   - Download the 512-D embedding model directly (~174 MB):
     ```powershell
     curl -L -o models/adaface_ir50_base.onnx https://huggingface.co/globalnebula/adaface-ir50-ms1mv2-onnx/resolve/main/adaface_ir50_ms1mv2.onnx
     ```

3. **Facial Aligner (`dfa_mobilenet_aligner.onnx`)**:
   - Built from CVLFace MobileNet landmark aligner. Run the exporter to generate `models/dfa_mobilenet_aligner.onnx`:
     ```powershell
     python scripts/download_open_models.py
     ```

#### Verify Model Setup
Check that all 3 ONNX models load properly:
```powershell
python scripts/download_open_models.py
```
You should see:
```text
Validated yolov8n-face.onnx:       Input: ['batch', 3, 'height', 'width']  Output: ['batch', ...]
Validated adaface_ir50_base.onnx:  Input: ['batch', 3, 112, 112]          Output: ['batch', 512]
Validated dfa_mobilenet_aligner.onnx: Input: ['batch_size', 3, 160, 160]   Output: ldmk, bbox, score
```

---

### 4. Environment Variables (`.env`)
Create a `.env` file in the project root:
```ini
detector_model="yolov8n-face.onnx"
head_model=""
recognizer_model="adaface_ir50_base.onnx"
reid_model=""
recognition_threshold_override=0.20
secret_key="your_secure_random_key_here"
```

### 5. Seed Cameras & Launch Server
```powershell
# Register default cameras in the SQLite database
python scripts/seed_cameras.py

# Launch the FastAPI web server and camera workers
python scripts/run.py
```
Open **`http://127.0.0.1:8000`** in your browser.
* **Default Credentials**: `inomjon` / `123456`

---

## Web Navigation Guide

| Route | Page | Purpose |
|---|---|---|
| `/` | **Dashboard** | Real-time transit events, present employee stats, and system KPIs. |
| `/live` | **Live Camera Console** | Real-time WebSocket video streaming with bounding boxes and track states. |
| `/employees` | **Employee Roster** | List, search, view attendance history, or delete employees. |
| `/employees/add` | **Enrollment** | Register new employees using multi-angle photos or webcam captures. |
| `/attendance` | **Attendance Logs** | Daily attendance records, transition timestamps, and CSV export. |
| `/gallery/review` | **Gallery Enrichment** | Human-in-the-loop review to approve corridor frames and inspect lookalike pairs. |
| `/cameras` | **Camera Config** | Manage RTSP streams, camera roles (`IN`, `OUT`, `BOTH`), and status. |

---

## Useful CLI Utilities

* **Profile Pipeline Latency & Jitter**:
  ```powershell
  python bench/profile_models.py
  ```
* **Benchmark Live Video / Replay Stream ($N$ frames)**:
  ```powershell
  python scripts/live_test.py 60
  ```
* **Calibrate Camera Virtual Tripwire Line**:
  ```powershell
  python scripts/set_direction.py --camera 1 --line 0.50,0.0,0.50,1.0 --inside right --depth grow
  ```
* **Extract Diverse Training Frames from Portrait Video**:
  ```powershell
  python scripts/extract_enrollment_frames.py --video sample.mp4 --out face_id_users/1_Name/ --count 5
  ```
* **Reset Attendance Records (Keep Employees & Cameras)**:
  ```powershell
  python scripts/reset_attendance.py --apply --media
  ```

---

## Architecture & Layout

```text
app/
  ├── api/              # FastAPI routers (auth, HTML pages, WebSocket live stream)
  ├── core/             # AI core (OnnxFaceDetector, FaceAligner, FaceRecognizer, ByteTrack)
  ├── db/               # SQLAlchemy models and SQLite session management
  ├── services/         # Enrollment, Attendance recording, CameraWorker pipeline
  └── config.py         # Global configuration settings and thresholds
bench/                  # Latency, FPS, and accuracy benchmarking scripts
data/                   # Local databases, direction previews, and debug captures (gitignored)
media/                  # Attendance snapshot JPEGs and employee photos (gitignored)
models/                 # Runtime ONNX neural network weights (gitignored)
scripts/                # Administrative CLI utilities (run, enroll, test, set_direction)
templates/              # Responsive Jinja2 Bootstrap HTML templates (100% English)
static/                 # Client-side JavaScript, CSS, and icons
```

---

## License & Credits
Developed as an enterprise-grade ML attendance system. Neural models based on YOLOv8 (Ultralytics), DFA MobileNet (CVLFace), and AdaFace (IR-50 base).
