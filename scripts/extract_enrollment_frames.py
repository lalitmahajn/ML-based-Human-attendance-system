#!/usr/bin/env python3
"""Extract sharp, diverse face frames from a video for employee enrollment,
with optional direct enrollment into the attendance database.

Usage:
    # 1. Just extract photos to review:
    python scripts/extract_enrollment_frames.py --video path/to/video.mp4 --name "Person1" --rotate 90 --count 6

    # 2. Extract and directly enroll into the system:
    python scripts/extract_enrollment_frames.py --video path/to/video.mp4 --name "Person1" --rotate 90 --enroll
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.quality import estimate_pose, sharpness_of
from app.services.enrollment import (
    Enroller,
    EnrollmentImage,
    enroll_employee_captures,
    load_gallery,
)
from app.db.session import init_db

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def rotate_frame(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def main():
    parser = argparse.ArgumentParser(description="Extract clean enrollment face frames from video.")
    parser.add_argument("--video", required=True, help="Path to input video file (e.g. .mp4, .mov)")
    parser.add_argument("--name", required=True, help="Person name (e.g. 'John Doe')")
    parser.add_argument("--count", type=int, default=6, help="Target number of frames to extract (default: 6)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Rotate frames if video is sideways (0, 90, 180, 270)")
    parser.add_argument("--out-dir", default="", help="Optional custom output directory")
    parser.add_argument("--enroll", action="store_true", help="Automatically enroll extracted frames into the DB")
    parser.add_argument("--department", default="", help="Optional department (for enrollment)")
    parser.add_argument("--position", default="", help="Optional position (for enrollment)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        log.error("Video file does not exist: %s", video_path)
        sys.exit(1)

    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.name).strip("_")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / "enrollment_frames" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    log.info("Opening %s: ~%d frames at %.1f fps", video_path.name, total_frames, fps)

    detector = build_detector(
        settings.detector_kind,
        settings.model_path(settings.detector_model),
        imgsz=settings.detect_width,
        conf=settings.detect_conf,
    )
    aligner = FaceAligner(settings.model_path(settings.aligner_model))

    candidates = []
    frame_idx = 0
    # Sample every ~0.15s (about 9-10 frames at 60 fps)
    step = max(1, int(fps * 0.15))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        if args.rotate:
            frame = rotate_frame(frame, args.rotate)

        # Detect face
        dets = detector.detect(frame)
        if not dets:
            continue

        # Take largest detected face
        best_det = max(dets, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        aligned = aligner.align(rgb, [best_det.box])
        if not aligned:
            continue

        ac = aligned[0]
        if ac.score < 0.85:
            continue

        yaw, pitch, roll = estimate_pose(ac.landmarks)
        sharp = sharpness_of(ac.aligned)
        face_px = float(max(best_det.box[2] - best_det.box[0], best_det.box[3] - best_det.box[1]))

        # Quality score combines alignment score, frontality/pose, and sharpness
        quality_score = ac.score * max(0.2, 1.0 - abs(pitch) / 45.0) * min(sharp / 250.0, 1.5)

        candidates.append({
            "frame": frame.copy(),
            "frame_idx": frame_idx,
            "yaw": yaw,
            "pitch": pitch,
            "sharpness": sharp,
            "face_px": face_px,
            "align_score": ac.score,
            "quality_score": quality_score,
        })

    cap.release()

    if not candidates:
        log.warning("No clear faces found in video. If the video is rotated, check --rotate.")
        sys.exit(1)

    log.info("Found %d candidate face frames across the video.", len(candidates))

    # Bin by yaw angles (center/frontal, left, right) to ensure pose diversity
    center = [c for c in candidates if abs(c["yaw"]) <= 12]
    left = [c for c in candidates if c["yaw"] < -12]
    right = [c for c in candidates if c["yaw"] > 12]

    center.sort(key=lambda c: c["quality_score"], reverse=True)
    left.sort(key=lambda c: c["quality_score"], reverse=True)
    right.sort(key=lambda c: c["quality_score"], reverse=True)

    selected = []
    # Pick top frontal, top left, top right
    if center:
        selected.extend(center[:max(2, args.count // 2)])
    if left:
        selected.extend(left[:max(1, (args.count - len(selected)) // 2)])
    if right:
        selected.extend(right[:max(1, args.count - len(selected))])

    # Fill remaining from top candidates overall
    all_sorted = sorted(candidates, key=lambda c: c["quality_score"], reverse=True)
    for c in all_sorted:
        if len(selected) >= args.count:
            break
        if not any(c["frame_idx"] == s["frame_idx"] for s in selected):
            selected.append(c)

    saved_paths = []
    captures_for_enrollment = []
    for i, item in enumerate(selected, 1):
        filename = f"{safe_name}_{i:02d}_yaw{int(item['yaw']):+d}.jpg"
        filepath = out_dir / filename
        cv2.imwrite(str(filepath), item["frame"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(filepath)
        captures_for_enrollment.append(
            EnrollmentImage(source_file=filepath.name, image_bgr=item["frame"])
        )
        log.info("  Saved: %s (face=%dpx, yaw=%+d°, pitch=%+d°, sharpness=%.1f, align=%.3f)",
                 filepath.name, int(item["face_px"]), int(item["yaw"]), int(item["pitch"]),
                 item["sharpness"], item["align_score"])

    log.info("\nExtracted %d high-quality enrollment photos into:", len(saved_paths))
    log.info("  %s", out_dir)

    if args.enroll:
        log.info("\nEnrolling '%s' into attendance gallery...", args.name)
        init_db()
        enroller = Enroller(detector=detector, aligner=aligner)
        try:
            res = enroll_employee_captures(
                full_name=args.name,
                position=args.position,
                department=args.department,
                phone_number="",
                captures=captures_for_enrollment,
                enroller=enroller,
            )
            log.info("SUCCESS: Enrolled '%s' (ID=%d, External=%s, Embeddings=%d)",
                     args.name, res.employee_id, res.external_id, res.embeddings)
            g = load_gallery()
            log.info("Gallery updated: %d embeddings / %d people now in memory.", len(g), g.n_people)
        except Exception as exc:
            log.error("Enrollment failed: %s", exc)
    else:
        log.info("\nTo enroll these photos automatically into the gallery:")
        log.info('  python scripts/extract_enrollment_frames.py --video "%s" --name "%s" --rotate %d --enroll',
                 args.video, args.name, args.rotate)
        log.info("Or upload them manually via http://127.0.0.1:8000/employees/add")


if __name__ == "__main__":
    main()
