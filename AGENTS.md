# Project Knowledge & Agent Guidelines (Attendance System)

> **CRITICAL FOR ALL CODING AGENTS**: Read this file before reading any legacy files in `docs/`. This file captures the active environment, active model decisions, hardware capabilities, and architectural history so you do not repeat past mistakes or get misled by stale historical notes.

---

## 1. Active Hardware & Environment

- **Host OS**: Windows 11 (PowerShell / pwsh)
- **Python Environment**: `venv\Scripts\python.exe` (Python 3.11.9, NOT Linux Conda).
- **GPU**: **NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM)**.
  - Architecture: **Blackwell (`sm_120`)**, CUDA UMD 13.4, Driver 616.x.
  - CUDA Toolkit: Installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`.
- **Legacy Docs Warning**: Historical docs in `docs/OPERATIONS.md`, `docs/BENCHMARKS.md`, `docs/ARCHITECTURE.md`, etc., refer to the original author's legacy setup (Ubuntu Linux, RTX 3070 Laptop, Python 3.10 conda "yolo"). **DO NOT** assume Linux paths, `.so` files, or an RTX 3070.

---

## 2. Active Neural Models vs. Original Author's Private Models

> [!CAUTION]
> **DO NOT USE OR REVERT TO THE ORIGINAL AUTHOR'S PROPRIETARY MODELS.**
> The original author's models referenced in `docs/` and default `app/config.py` were **private, licensed, or missing files** that do not exist in this repository. Attempting to use them will crash the pipeline with `FileNotFoundError`.

### Model Comparison & Rationale Table

| Pipeline Component | Original Author's Model (DO NOT USE) | Active Replacement Model (IN USE) | Why We Changed It |
|---|---|---|---|
| **Face Recognizer** | `adaface_ir101_finetune_fp16.onnx` (or `.enc`) | `models/adaface_ir50_base.onnx` (174 MB) | The author's IR-101 was a private NIST FRVT fine-tuned checkpoint tied to proprietary license encryption (`.enc`). It does not exist in `models/`. We replaced it with open-source AdaFace IR-50 base and configured `recognition_threshold_override=0.20` in `.env`. |
| **Head / Body Detector** | `yolov8n_head_960x544_fp16.onnx` | `head_model=""` (Disabled in `.env`) | The author's custom CrowdHuman head export does not exist in `models/`. Setting `head_model=""` tells the pipeline to track directly on face boxes via `yolov8n-face.pt`. |
| **Face Detector** | `yolov8n-face.pt` | `models/yolov8n-face.onnx` (11.8 MB) | Exported to ONNX. Runs accelerated on RTX 5050 GPU via `CUDAExecutionProvider` through `OnnxFaceDetector` in `app/core/detector.py` (falls back to `.pt` on CPU if ONNX is missing). |
| **Face Aligner** | `dfa_mobilenet_aligner.onnx` | `models/dfa_mobilenet_aligner.onnx` (2 MB) | Kept in use. Accelerated on RTX 5050 GPU via `CUDAExecutionProvider`. |
| **Person ReID** | `reid_osnet_x0_75_256x128_e512_fp16.onnx` | `reid_model=""` (Disabled in `.env`) | Disabled in `.env` to ensure zero unnecessary VRAM overhead during core attendance operations. |

### Strict `.env` Model Rules for Agents
Never modify `.env` to point to the author's missing models. The active `.env` must remain:
```ini
head_model=""
recognizer_model="adaface_ir50_base.onnx"
reid_model=""
recognition_threshold_override=0.20
```

---

## 3. Key Technical Decisions & Why Changes Were Made

### A. Windows DLL Preloading (`app/core/onnx_env.py`)
- **Problem**: In Python 3.8+ on Windows, Python does not automatically resolve DLLs from PATH for C extensions. `onnxruntime-gpu` failed to load `cublasLt64_12.dll` and `cudnn64_9.dll`.
- **Solution**: `preload_cuda_libs()` in `app/core/onnx_env.py` uses `os.add_dll_directory()` pointing to `venv/Lib/site-packages/torch/lib/` (where CUDA 12 and cuDNN 9 DLLs reside).
- **Rule**: Never remove the Windows DLL directory loading logic in `onnx_env.py`.

### B. Prevention of `onnxruntime` CPU Package Shadowing
- **Problem**: `onnxruntime` (CPU) and `onnxruntime-gpu` provide the same module namespace. If CPU `onnxruntime` is installed, it silently overrides GPU execution, forcing the entire pipeline to CPU.
- **Rule**: Only `onnxruntime-gpu` (v1.23.2 or newer) must be installed. Never install plain `onnxruntime`.

### C. 100% GPU Execution via ONNX Runtime & Blackwell (`sm_120`)
- **Problem**: PyTorch Windows wheels currently only bundle binary kernels up to `sm_90`. Calling PyTorch CUDA operations on the RTX 5050 (Blackwell `sm_120`) crashes with `CUDA error: no kernel image is available for execution on the device`.
- **Solution**: We exported `yolov8n-face.pt` to `models/yolov8n-face.onnx` and implemented `OnnxFaceDetector` in `app/core/detector.py`. ONNX Runtime with `CUDAExecutionProvider` uses JIT/PTX to natively execute YOLO detection on the RTX 5050 GPU (reducing detection latency from 137 ms to 20 ms). All 3 neural models (Detector, Aligner, Recognizer) now run 100% on the GPU.

### D. Full English UI & Backend Localization
- **Rule**: All HTML templates (`templates/**/*.html`), client scripts (`static/js/main.js`), and server-side responses (`app/api/pages.py`, validation messages, flash notices) have been translated 100% line-by-line from Uzbek to English.
- **NEVER** re-introduce Uzbek strings into any template, JavaScript file, or API error response.

---

## 4. Operational Commands

- **Run Server**: `venv\Scripts\python.exe scripts\run.py` (runs on `http://127.0.0.1:8000`).
- **Enrollment Web Route**: `http://127.0.0.1:8000/employees/add`
- **Default Credentials**: `inomjon` / `123456`
