# Operations

## Environment

The runtime needs **no PyTorch for inference** — the aligner and recognizer are
ONNX. PyTorch is present only because YOLOv8n-face runs through Ultralytics.

> [!IMPORTANT]
> **Active Production Setup**: See [AGENTS.md](file:///c:/Users/lalit/Downloads/Project/attendance/AGENTS.md) for the authoritative active configuration.
> - **OS / Hardware**: Windows 11, **NVIDIA GeForce RTX 5050 (8GB VRAM, sm_120 Blackwell)**, CUDA 13.3/13.4.
> - **Python**: 3.11 (`venv\Scripts\python.exe`).
> - **Active Models**: `yolov8n-face.pt`, `dfa_mobilenet_aligner.onnx` (GPU CUDA), `adaface_ir50_base.onnx` (GPU CUDA).
> - The notes below refer to the original developer's historical Linux/RTX 3070 benchmarking environment and are retained for reference.

```
(Historical reference only - see AGENTS.md for active configuration)
Python 3.10.16      conda env "yolo"
onnxruntime-gpu     1.23.2   → TensorRT, CUDA, CPU providers
CUDA 12.4 / cuDNN 9 supplied by torch's bundled nvidia-* wheels
GPU                 RTX 3070 Laptop 8 GB (compute 8.6)
```

> **`onnxruntime` and `onnxruntime-gpu` must never both be installed.** They ship
> the same `onnxruntime` module; the CPU package shadows the GPU one and
> `get_available_providers()` silently returns CPU only. That was the state of
> this machine at the start. Fix:
> ```bash
> pip uninstall -y onnxruntime onnxruntime-gpu
> pip install onnxruntime-gpu==1.23.2
> python -c "import onnxruntime as o; print(o.get_available_providers())"
> ```
> Expect `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`.
>
> **This regresses every time a model is exported.** `ultralytics` treats
> `onnxruntime` as an export requirement and pip-installs the CPU package
> mid-run ("AutoUpdate success ✅ ... installed 1 package: ['onnxruntime']"),
> silently re-shadowing the GPU build. It has happened five times so far. After
> *any* `YOLO(...).export(...)`, re-run the fix above — use `--no-deps` to keep
> pip from dragging the CPU wheel back in:
> ```bash
> pip uninstall -y onnxruntime onnxruntime-gpu
> pip install --no-deps onnxruntime-gpu==1.23.2
> pip list | grep -E "^onnxruntime"     # must show onnxruntime-gpu ONLY
> ```

Other dependencies: `sqlalchemy>=2.0 aiosqlite fastapi uvicorn jinja2
python-multipart pydantic-settings opencv-python ultralytics lap`.

## Running

```bash
python scripts/enroll.py         # rebuild gallery (idempotent; wipes embeddings first)
CAMERA_USER=admin CAMERA_PASSWORD='...' python scripts/seed_cameras.py   # register cameras + roles
python scripts/run.py            # workers + web UI  → http://localhost:8000
```

Reach it from another machine on the LAN at `http://192.168.0.186:8000`.

| endpoint | purpose |
|---|---|
| `/` | dashboard: today's attendance, camera health, recent events |
| `/live` | annotated MJPEG per camera |
| `/api/health` | per-camera stream state, fps, throughput — use for monitoring |
| `/api/attendance?day=YYYY-MM-DD` | JSON |
| `/api/attendance/export` | CSV |
| `POST /api/gallery/reload` | re-read the gallery after enrolling someone new |
| `/api/debug/captures` | what the debug folder currently holds, per person |

## Visual review of recognitions

Every committed recognition writes four files into `data/debug/<person>/`:

| file | what |
|---|---|
| `*_frame.jpg` | the whole annotated frame — context, who else was around |
| `*_face.jpg` | the native-resolution crop the aligner was given |
| `*_aligned.jpg` | **the 112×112 the recognizer actually saw** |
| `*.json` | score, margin, pose, gate, camera, track, thresholds in force |

The aligned crop is the one that settles arguments: if a name is wrong, it shows
immediately whether the pipeline mis-framed the face or the gallery genuinely
holds a lookalike.

```bash
python scripts/export_debug.py         # backfill from the DB + build a contact sheet
xdg-open data/debug/index.html         # every recognition, grouped by person
```

The contact sheet outlines the aligned crops in teal so they are easy to pick
out. Each person keeps their newest `debug_max_per_person` (40) captures:
the count comes from what is on disk, so a restart or a second camera does
not reset it, and at the cap the oldest capture is deleted to make room. It
is a rolling window on purpose - this is the picture the events page shows
beside a recognition, so the one that must never be missing is the most
recent. Capture is skipped when free disk drops under
`save_all_min_free_gb`; set `debug_capture=false` to turn it off entirely.
A folder already far over the cap shrinks by one per new capture rather than
being trimmed at startup, because the surplus is evidence for days an
operator may still be reviewing.

## Diagnostics

```bash
python scripts/live_test.py 40      # per-stage timings against live cameras
python scripts/diagnose_live.py 60  # per-face: quality gates, match scores, crops
python bench/bench_models.py        # model speed on this GPU
python bench/validate_gallery.py    # gallery separation, FAR/FRR, rank-1
python bench/sweep_margin.py        # re-tune the aligner crop margin
python bench/test_attendance.py     # state machine over a simulated day
```

`diagnose_live.py` is the one to reach for when someone is not being recognized.
It prints, per face: pixel size, sharpness, yaw, pitch, which gate rejected it
(or `PASS`), the best gallery score and the runner-up — and writes the aligned
crops to `data/diagnose/` so you can look at what the model actually saw.

## Camera configuration

Both cameras are Hikvision DS-2CD2083G2-I. Applied via ISAPI:

| setting | value | why |
|---|---|---|
| codec | H.264 | cheaper decode, better ffmpeg/NVDEC support than H.265 |
| resolution | 3840×2160 | high mount + wide lens ⇒ resolution *is* face pixels |
| framerate | 12 | plenty through a doorway; halves decode |
| GOP | 12 | ~1 s keyframes, fast reconnect recovery |
| bitrate | 8192 kbps VBR | H.264 needs more bits than H.265 for equal quality |
| WDR | open, level 50 | was **off**; entrance windows were fully blown out |

```bash
# read current config
curl -s --digest -u admin:PASS http://192.168.1.2/ISAPI/Streaming/channels/101

# write (GET, edit, PUT the whole document — partial PUTs are ignored)
curl -s --digest -u admin:PASS -X PUT -H "Content-Type: application/xml" \
     --data-binary @new.xml http://192.168.1.2/ISAPI/Streaming/channels/101
```

> **Changing the codec resets the other encoder fields.** Set `videoCodecType`
> first, then re-GET and PUT framerate/GOP/bitrate in a *second* request. A
> combined PUT returns `statusString OK` and silently ignores them. Always
> re-read after ~20 s to confirm the change persisted — during this build the
> values reverted once after a network interruption.

## Tuning

Everything lives in `app/config.py`, overridable via `.env`.

| symptom | knob | direction |
|---|---|---|
| known people not recognized | `recognition_threshold` | lower (measure first) |
| wrong person identified | `recognition_threshold`, `second_best_margin` | raise |
| faces rejected as too small | `min_face_px`, `detect_width` | lower / raise |
| faces rejected as blurred | `min_laplacian_var` | lower, or cap camera shutter at 1/250 |
| identity commits too slowly | `vote_required` | lower (3→2, costs accuracy) |
| duplicate check-ins | `event_cooldown_s` | raise |
| CPU/GPU saturated | `process_every_nth` | raise (see note below) |

**Do not tune the threshold by intuition.** Run `diagnose_live.py`, look at the
real score distribution, then choose. See "Calibration" below.

### Before raising `process_every_nth`, check what is actually saturated

Raising it is the wrong first move if the cost is CPU preprocessing rather than
model inference — and on this host it was. `detect` reported 42–68 ms while the
GPU inside it took 12.5 ms; the rest was `cv2.dnn.blobFromImage` allocating an
11 MB buffer per frame against ~1 GB of free RAM. Skipping frames hides that
instead of fixing it, and costs trajectory points that direction resolution
needs.

Diagnose before tuning:

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv   # GPU actually busy?
cat /proc/loadavg && free -g                          # CPU / RAM pressure
```

Low GPU utilisation with a high load average means the bottleneck is CPU-side
buffer work, not the network. Sessions built through `best_providers()` (every
session in `app/`) preload the CUDA libraries themselves; a bench script that
constructs `ort.InferenceSession` directly must call `preload_cuda_libs()`
first or it silently falls back to CPU and every measurement is meaningless —
check `session.get_providers()` before trusting a benchmark.

## Calibration — done, and how to redo it

Calibrated 2026-08-19 from two real walkthroughs. Full numbers in
`docs/BENCHMARKS.md`.

| | value | evidence |
|---|---|---|
| `recognition_threshold` | **0.26** | impostor ceiling 0.184 over 43 live observations; genuine reaches 0.514 |
| `second_best_margin` | 0.08 | one live event landed at margin 0.065 |
| quality gates | loosened | `degrade_test.py`: recognition survives 60 px, sharpness 47, 35° pitch |
| camera shutter | 1/250 | was **1/25** — the single largest accuracy fix found |

Redo it after moving a camera, changing the lens, or swapping the recognizer:

```bash
python scripts/walkthrough.py 240      # once with a NON-enrolled person
python scripts/walkthrough.py 240      # once with enrolled people
python bench/calibrate_threshold.py    # reads both logs, recommends a threshold
```

Pick the highest threshold that still commits the genuine tracks with zero false
commits. Do not pick it by intuition — the gallery-photo measurement suggested
0.40, and live crops needed 0.26.

## Closing the domain gap

Live scores peak near 0.51; gallery-to-gallery scores sit near 0.90. The
difference is domain — clean enrolment photos vs corridor crops — not model
quality. Adding confirmed corridor crops to the gallery is the highest-value
accuracy work left:

```bash
python scripts/enroll_from_live.py 300     # collect high-confidence crops
#  review data/live_enroll/<person>/ and DELETE anything that is the wrong person
python scripts/enroll_from_live.py --commit
curl -X POST http://localhost:8000/api/gallery/reload
```

The review step is not optional. Auto-enrolling whatever the matcher believes
would let one wrong match poison an identity permanently.

## Deployment to another machine

`models/` holds real weight files (225 MB), not symlinks, with
`MANIFEST.sha256`. Verify after copying:

```bash
cd models && sha256sum -c MANIFEST.sha256
```

The gallery lives in `data/ematsy.db`; rebuild it anywhere with
`scripts/enroll.py` from `face_id_users/`. Nothing depends on paths outside the
project directory.

## Known operational limits

- **SQLite.** Fine for two cameras. Move to PostgreSQL before adding a third or
  serving reports to many users; `UtcDateTime` already behaves correctly there.
- **Authentication is a session cookie** (`app/api/auth.py`): admin, operator
  and viewer roles, a lockout after repeated failed logins, and a cross-site
  check on every state-changing request. That check reads the browser's own
  `Sec-Fetch-Site` header, because comparing the request's `Origin` against
  its `Host` is wrong behind a proxy - the proxy rewrites `Host`, and the
  operator's own form then looks like a stranger. A browser older than
  Chrome 76 / Firefox 90 / Safari 16.4 sends no such header and falls back to
  that comparison; set `trusted_hosts=aiscan.airi.uz` in `.env` if one has to
  be supported. It is defence in depth behind the `SameSite=Lax` cookie, not
  a CSRF token: a page on the same origin - the sibling apps on this host -
  is indistinguishable from ours by any header. Serve it behind TLS (the
  cookie is only marked `Secure` when the request arrived over https).
- **No liveness detection.** Deliberate — these cameras are 3 m up behind a wide
  lens, where holding a photo to the lens is not practical. Revisit if a
  door-level camera is added.
- **Disk at 97%** (27 GB free). Snapshots accumulate in `media/snapshots/`;
  `snapshot_retention_days` is set to 90 but the cleanup job is not yet wired.
- **RAM 13 GB, ~2 GB free with Chrome open.** 4K frames are 24 MB each; queues
  are bounded at 2 per camera, but close browsers when running.
