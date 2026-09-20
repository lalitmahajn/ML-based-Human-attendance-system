# Gemini / AI Assistant Instructions

Please refer to [AGENTS.md](file:///c:/Users/lalit/Downloads/Project/attendance/AGENTS.md) for:
1. **Model Replacements**: We are **NOT** using the original author's proprietary/missing models (`adaface_ir101_finetune_fp16.onnx`, `yolov8n_head_960x544_fp16.onnx`). We are using the public open-source `adaface_ir50_base.onnx` with `recognition_threshold_override=0.20` and tracking on face boxes (`head_model=""`). Never revert `.env` to the missing author models.
2. **Active Hardware Configuration**: Windows 11, NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM, Blackwell `sm_120`), CUDA 13.3/13.4, Python 3.11 `venv`.
3. **Blackwell GPU Compatibility Rules**: ONNX Runtime accelerates detector (`yolov8n-face.onnx`), aligner (`dfa_mobilenet_aligner.onnx`), and recognizer (`adaface_ir50_base.onnx`) 100% on RTX 5050 GPU via `CUDAExecutionProvider`.
4. **English Localization**: 100% English UI and backend responses. Never reintroduce Uzbek text.

Always consult [AGENTS.md](file:///c:/Users/lalit/Downloads/Project/attendance/AGENTS.md) before consulting any legacy historical notes in `docs/`.
