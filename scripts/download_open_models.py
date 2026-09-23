import os
import sys
import urllib.request
from pathlib import Path

vendor_dir = Path(__file__).resolve().parent / "vendor_downloads"
cvlface_dir = vendor_dir / "CVLface"
sys.path.insert(0, str(cvlface_dir))

import torch
import onnx
import onnxruntime
from huggingface_hub import hf_hub_download

models_dir = Path(__file__).resolve().parent.parent / "models"
models_dir.mkdir(exist_ok=True)

def download_file(url, out_path):
    if not out_path.exists():
        print(f"Downloading {out_path.name}...")
        urllib.request.urlretrieve(url, out_path)
    else:
        print(f"{out_path.name} already exists.")

def main():
    yolo_path = models_dir / "yolov8n-face.pt"
    download_file("https://huggingface.co/junjiang/GestureFace/resolve/main/yolov8n-face.pt", yolo_path)

    yolo_onnx_path = models_dir / "yolov8n-face.onnx"
    if not yolo_onnx_path.exists():
        print("Exporting YOLOv8n-face to ONNX (dynamic=True)...")
        from ultralytics import YOLO
        yolo_model = YOLO(str(yolo_path))
        exported_path = yolo_model.export(format="onnx", imgsz=640, dynamic=True)
        if Path(exported_path) != yolo_onnx_path and Path(exported_path).exists():
            import shutil
            shutil.move(str(exported_path), str(yolo_onnx_path))
        print("YOLOv8n-face exported to ONNX successfully.")
    else:
        print("yolov8n-face.onnx already exists.")

    adaface_path = models_dir / "adaface_ir50_base.onnx"
    download_file("https://huggingface.co/globalnebula/adaface-ir50-ms1mv2-onnx/resolve/main/adaface_ir50_ms1mv2.onnx", adaface_path)

    dfa_onnx_path = models_dir / "dfa_mobilenet_aligner.onnx"
    if not dfa_onnx_path.exists():
        print("Exporting CVLFace DFA Mobilenet to ONNX...")
        try:
            if not cvlface_dir.exists():
                import subprocess
                print("Cloning CVLface vendor repo for DFA aligner export...")
                vendor_dir.mkdir(parents=True, exist_ok=True)
                subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/mk-minchul/CVLface.git", str(cvlface_dir)])

            import importlib
            aligners_mod = importlib.import_module("cvlface.research.recognition.code.run_v1.aligners")
            get_aligner = aligners_mod.get_aligner
            from omegaconf import OmegaConf
            
            model_safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
            
            cfg = OmegaConf.create({
                "name": "differentiable_face_aligner",
                "arch": "mobile0.25",
                "input_size": 160,
                "output_size": 112,
                "input_padding_ratio": 0.0,
                "input_padding_val": "zero",
                "freeze": True,
                "start_from": ""
            })
            model = get_aligner(cfg)
            
            from safetensors.torch import load_file
            state_dict = load_file(model_safetensors_path)
            # The safetensors from huggingface has prefix 'model.' which maps to the 'net.' prefix in DifferentiableFaceAligner?
            # Wait, the HF model wrapper does `self.model = get_aligner()`. `self.model.load_state_dict()` works because the safetensors keys are likely `model.net.XXX` or `net.XXX`.
            # Let's inspect the keys before loading.
            
            # Remove 'model.' prefix if present at the top level
            if any(k.startswith("model.net.") for k in state_dict.keys()):
                state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
            elif any(k.startswith("net.") for k in state_dict.keys()):
                pass # it's fine
            else:
                # if it just starts with stage1, etc. we might need to prefix it with 'net.'
                if list(state_dict.keys())[0].startswith("model."):
                    state_dict = {k.replace("model.", "net."): v for k, v in state_dict.items()}
                else:
                    state_dict = {"net." + k: v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            dummy_input = torch.randn(1, 3, 160, 160)
            
            # In cvlface DifferentiableFaceAligner, forward returns:
            # aligned_x, orig_pred_ldmks, aligned_ldmks, score, thetas, normalized_bbox
            # The ONNX expected outputs are (ldmk, bbox, score, resized) in attendance repo!
            # Wait, the attendance aligner.py expects: ldmk, _bbox, score, resized = session.run
            # Let's write a simple wrapper that outputs exactly what's needed.
            class AlignerWrapper(torch.nn.Module):
                def __init__(self, dfa_model):
                    super().__init__()
                    self.dfa = dfa_model
                    self.net = dfa_model.net
                def forward(self, x):
                    # RetinaFace returns (bbox_regressions, classifications, ldm_regressions, agg, theta)
                    tensor = self.net(x, self.dfa.prior_box)[3]
                    bbox, score, ldmk = torch.split(tensor, [4, 2, 10], dim=1)
                    score = torch.softmax(score, dim=-1)[:, 1:2]
                    ldmk = ldmk.view(-1, 5, 2)
                    normalized_bbox = bbox / 160.0
                    
                    # Dummy resized tensor to satisfy 4-output signature, unused in 'sharp' mode
                    resized = torch.zeros(x.size(0), 3, 112, 112, dtype=x.dtype, device=x.device)
                    return ldmk, normalized_bbox, score, resized

            wrapper_model = AlignerWrapper(model)
            
            torch.onnx.export(
                wrapper_model, 
                dummy_input, 
                str(dfa_onnx_path), 
                input_names=["image"], 
                output_names=["ldmk", "bbox", "score", "resized"], 
                opset_version=16,
                dynamic_axes={"image": {0: "batch_size"}, "ldmk": {0: "batch_size"}, "bbox": {0: "batch_size"}, "score": {0: "batch_size"}, "resized": {0: "batch_size"}}
            )
            print("CVLFace DFA Mobilenet exported successfully.")
        except Exception as e:
            print(f"Error exporting CVLFace DFA: {e}")
            import traceback
            traceback.print_exc()

    for onnx_file in [yolo_onnx_path, adaface_path, dfa_onnx_path]:
        if onnx_file.exists():
            try:
                session = onnxruntime.InferenceSession(str(onnx_file), providers=['CPUExecutionProvider'])
                print(f"Validated {onnx_file.name}:")
                for idx, inp in enumerate(session.get_inputs()):
                    print(f"  Input {idx}: {inp.name} {inp.shape}")
                for idx, out in enumerate(session.get_outputs()):
                    print(f"  Output {idx}: {out.name} {out.shape}")
            except Exception as e:
                print(f"Error validating {onnx_file.name}: {e}")

if __name__ == "__main__":
    main()
