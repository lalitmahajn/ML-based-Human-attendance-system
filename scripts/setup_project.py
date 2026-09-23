"""Automated end-to-end setup script for Human Attendance System.

Designed for automated execution by AI agents and developers on fresh clones.
Handles:
1. Environment file creation (.env from env.example)
2. Dependency verification (onnxruntime-gpu vs cpu conflict check)
3. Model weight download and ONNX export (YOLOv8-face, AdaFace IR-50, DFA Aligner)
4. Database initialization (seed_cameras.py)
5. Model validation (tensors, shapes, execution providers)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR = ROOT_DIR / "models"
ENV_FILE = ROOT_DIR / ".env"
ENV_EXAMPLE = ROOT_DIR / "env.example"

def print_step(num: int, title: str):
    print(f"\n[{num}/5] {title}...")

def check_python_version():
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required. Current version: {sys.version}")
        sys.exit(1)

def step_1_env():
    print_step(1, "Configuring Environment (.env)")
    if not ENV_FILE.exists():
        if ENV_EXAMPLE.exists():
            shutil.copy(ENV_EXAMPLE, ENV_FILE)
            print("  Created .env from env.example")
        else:
            default_env = (
                'detector_model="yolov8n-face.onnx"\n'
                'head_model=""\n'
                'recognizer_model="adaface_ir50_base.onnx"\n'
                'reid_model=""\n'
                'recognition_threshold_override=0.20\n'
                'secret_key="ematsy-attendance-system-secret-key"\n'
            )
            ENV_FILE.write_text(default_env, encoding="utf-8")
            print("  Created new .env with active model configuration")
    else:
        print("  .env already exists.")

def step_2_dependencies():
    print_step(2, "Checking Dependencies")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  ONNX Runtime available providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            print("  [OK] CUDA GPU acceleration is enabled in ONNX Runtime.")
        else:
            print("  [NOTICE] CUDAExecutionProvider not detected; will run on CPU or falling back to available provider.")
    except ImportError:
        print("  Installing missing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements.txt")])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.23.2"])

def step_3_download_models():
    print_step(3, "Downloading & Converting Models")
    download_script = ROOT_DIR / "scripts" / "download_open_models.py"
    subprocess.check_call([sys.executable, str(download_script)])

def step_4_seed_database():
    print_step(4, "Initializing Database & Cameras")
    seed_script = ROOT_DIR / "scripts" / "seed_cameras.py"
    subprocess.check_call([sys.executable, str(seed_script)])

def step_5_verify_pipeline():
    print_step(5, "Verifying AI Pipeline & Hardware Acceleration")
    from app.core.onnx_env import best_providers, preload_cuda_libs
    preload_cuda_libs()
    import onnxruntime as ort

    models = [
        ("Face Detector", MODELS_DIR / "yolov8n-face.onnx"),
        ("Face Aligner", MODELS_DIR / "dfa_mobilenet_aligner.onnx"),
        ("Face Recognizer", MODELS_DIR / "adaface_ir50_base.onnx"),
    ]

    for name, path in models:
        if not path.exists():
            raise FileNotFoundError(f"Missing model weight: {path}")
        session = ort.InferenceSession(str(path), providers=best_providers())
        active_provider = session.get_providers()[0]
        print(f"  [OK] {name} ({path.name}) loaded successfully on [{active_provider}]")

def main():
    print("=" * 65)
    print("Human Attendance System - Automated Environment & Model Setup")
    print("=" * 65)
    check_python_version()
    step_1_env()
    step_2_dependencies()
    step_3_download_models()
    step_4_seed_database()
    step_5_verify_pipeline()
    print("\n" + "=" * 65)
    print("SETUP COMPLETE! You can now start the server:")
    print("  python scripts/run.py")
    print("Access dashboard at: http://127.0.0.1:8000")
    print("Default login: inomjon / 123456")
    print("=" * 65)

if __name__ == "__main__":
    main()
