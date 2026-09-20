import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.onnx_env import preload_cuda_libs
preload_cuda_libs()
import time, cv2, numpy as np
from app.services.enrollment import load_gallery
from app.core.pipeline import CameraPipeline, Frame
import onnxruntime as ort

gallery = load_gallery()
pipe = CameraPipeline("bench", gallery)

# Read 100 frames from the replay video
cap = cv2.VideoCapture("data/recordings/Phone Camera/video_feed.mp4")
frames = []
for _ in range(100):
    ok, img = cap.read()
    if not ok:
        break
    frames.append(img)
cap.release()

print(f"Collected {len(frames)} test frames for stability benchmark.")

# Warmup (5 iterations)
for i in range(5):
    pipe.process(Frame(frames[i], time.time(), i))

# Profile all frames
timings = {"detect": [], "align": [], "embed": [], "total": []}
for i in range(len(frames)):
    f = Frame(frames[i], time.time(), i)
    res = pipe.process(f)
    for k in ("detect", "align", "embed", "total"):
        if k in res.timings:
            timings[k].append(res.timings[k])

print("\n=== MODEL PROCESSING LATENCY PROFILE (100 FRAMES) ===")
header = f"{'Stage':<12} {'Min':>8} {'Mean':>8} {'p50':>8} {'p95':>8} {'Max':>8} {'StdDev':>8} (all in ms)"
print(header)
print("-" * len(header))
for k in ("detect", "align", "embed", "total"):
    arr = np.array(timings[k])
    print(f"{k:<12} {arr.min():8.1f} {arr.mean():8.1f} {np.median(arr):8.1f} {np.percentile(arr, 95):8.1f} {arr.max():8.1f} {arr.std():8.2f}")

tot = np.array(timings["total"])
print(f"\nThroughput: {1000.0 / tot.mean():.1f} FPS (continuous capacity)")
print(f"Frame Time Budget for 20 FPS: 50.0 ms | Current p50: {np.median(tot):.1f} ms (Headroom: {50.0 - np.median(tot):.1f} ms)")

print("\n=== ACTIVE HARDWARE & PROVIDERS ===")
det_p = getattr(pipe.detector, "provider", "Unknown")
align_p = pipe.aligner.provider
rec_p = pipe.recognizer.session.get_providers()[0]
print(f"Face Detector : {det_p}")
print(f"Face Aligner  : {align_p}")
print(f"Face Recognizer: {rec_p}")
