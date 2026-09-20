import sys, os, time, cv2

url = sys.argv[1] if len(sys.argv) > 1 else "rtsp://192.168.31.164:8554/"
transport = sys.argv[2] if len(sys.argv) > 2 else "tcp"
print(f"Testing URL: {url} with transport {transport}", flush=True)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay"
params = [
    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000,
    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000
]
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
if not cap.isOpened():
    print("Failed to open capture!", flush=True)
    sys.exit(1)

print("Opened! Reading frame...", flush=True)
ret, frame = cap.read()
if ret and frame is not None:
    print(f"SUCCESS: Read frame shape={frame.shape}", flush=True)
else:
    print("Failed to read frame!", flush=True)
cap.release()
