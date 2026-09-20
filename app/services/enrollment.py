"""Build the local face gallery from the `face_id_users/` export.

Layout consumed:

    face_id_users/
      users.json                     master index
      NNN_<user_id>/metadata.json    per-person record
      NNN_<user_id>/image_01..05.png 1280x720 enrolment frames

One `FaceEmbedding` row per usable image — not an averaged centroid.  Matching
on max-similarity across a person's images handles pose and lighting spread
better than one mean vector, and it keeps a bad enrolment photo visible instead
of quietly dragging the centroid toward a different face.
"""
from __future__ import annotations

import base64
import binascii
from io import BytesIO
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from sqlalchemy import delete, or_, select

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.gallery import Gallery
from app.core.quality import assess, sharpness_of
from app.core.recognizer import FaceRecognizer
from app.db.models import Employee, FaceEmbedding
from app.db.session import session_scope
from app.services.augment import TAG

log = logging.getLogger(__name__)

MAX_BROWSER_CAPTURES = 32
MAX_CAPTURE_BYTES = 4 * 1024 * 1024
MAX_ENROLLMENT_BODY_BYTES = 10 * 1024 * 1024
MAX_ENROLLMENT_FIELDS = 12
MAX_ENROLLMENT_FIELD_BYTES = 64 * 1024
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_PIXELS = 12_000_000
UPLOAD_TYPES = {
    "image/jpeg": {".jpg", ".jpeg"},
    "image/png": {".png"},
    "image/webp": {".webp"},
}
CANONICAL_IMAGE_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
IMAGE_FORMATS = {
    "JPEG": ("image/jpeg", {".jpg", ".jpeg"}),
    "PNG": ("image/png", {".png"}),
    "WEBP": ("image/webp", {".webp"}),
}


class EnrollmentCaptureError(ValueError):
    """A browser capture could not safely produce a gallery embedding."""

    def __init__(self, message: str, rejections=()):
        super().__init__(message)
        self.rejections = tuple(rejections)


@dataclass(frozen=True)
class EnrollmentImage:
    source_file: str
    image_bgr: np.ndarray | None = None
    error: str = ""


@dataclass(frozen=True)
class EnrollmentRejection:
    source_file: str
    error: str


@dataclass(frozen=True)
class CaptureEmbedding:
    vector: np.ndarray
    quality: float


@dataclass(frozen=True)
class EnrollmentResult:
    employee_id: int
    external_id: str
    embeddings: int
    rejected: tuple[EnrollmentRejection, ...] = ()


def decode_browser_captures(captured_images: str, captured_image: str = "") -> list[EnrollmentImage]:
    """Decode the registration form's data URLs into OpenCV BGR frames."""
    values = None
    if captured_images:
        try:
            values = json.loads(captured_images)
        except (TypeError, json.JSONDecodeError) as exc:
            raise EnrollmentCaptureError(
                "Invalid face samples format. Please retake samples and submit again."
            ) from exc
        if not isinstance(values, list):
            raise EnrollmentCaptureError(
                "Face samples were not sent as a list. Please retake samples."
            )
    elif captured_image:
        values = [captured_image]
    else:
        values = []

    frames: list[EnrollmentImage] = []
    for index, value in enumerate(values, start=1):
        source_file = f"browser/{index:03d}.jpg"
        if index > MAX_BROWSER_CAPTURES:
            frames.append(EnrollmentImage(
                source_file,
                error=f"At most {MAX_BROWSER_CAPTURES} images can be submitted.",
            ))
            continue
        if not isinstance(value, str) or "," not in value:
            frames.append(EnrollmentImage(
                source_file, error="Image data is incomplete. Please retake sample."
            ))
            continue
        header, encoded = value.split(",", 1)
        mime_type = header.removeprefix("data:").split(";", 1)[0].lower()
        if mime_type not in UPLOAD_TYPES or ";base64" not in header:
            frames.append(EnrollmentImage(
                source_file, error="Image format not supported. Use JPEG, PNG, or WebP."
            ))
            continue
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            frames.append(EnrollmentImage(
                source_file, error="Image data is corrupted. Please retake sample."
            ))
            continue
        suffix = CANONICAL_IMAGE_SUFFIX[mime_type]
        source_file = f"browser/{index:03d}{suffix}"
        frames.append(_decode_image_payload(
            raw,
            source_file=source_file,
            filename=f"capture{suffix}",
            content_type=mime_type,
        ))
    return frames


def _signature_matches(raw: bytes, format_name: str) -> bool:
    if format_name == "JPEG":
        return raw.startswith(b"\xff\xd8\xff")
    if format_name == "PNG":
        return raw.startswith(b"\x89PNG\r\n\x1a\n")
    if format_name == "WEBP":
        return len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
    return False


def _decode_image_payload(
    raw: bytes,
    *,
    source_file: str,
    filename: str,
    content_type: str,
) -> EnrollmentImage:
    """Inspect a bounded image header before handing bytes to OpenCV."""
    if not raw or len(raw) > MAX_CAPTURE_BYTES:
        return EnrollmentImage(
            source_file,
            error="Image size exceeds 4 MB limit or file is empty.",
        )

    safe_name = Path(filename).name
    declared_mime = (content_type or "").split(";", 1)[0].lower()
    declared_suffix = Path(safe_name).suffix.lower()
    try:
        with Image.open(BytesIO(raw)) as image:
            actual_format = (image.format or "").upper()
            width, height = image.size
            if actual_format not in IMAGE_FORMATS or not _signature_matches(raw, actual_format):
                return EnrollmentImage(
                    source_file,
                    error=(
                        "Actual file format is not JPEG, PNG, or WebP. "
                        "Please select a valid image format."
                    ),
                )
            expected_mime, expected_suffixes = IMAGE_FORMATS[actual_format]
            if declared_mime != expected_mime or declared_suffix not in expected_suffixes:
                return EnrollmentImage(
                    source_file,
                    error=(
                        "File name or extension does not match its actual format. "
                        "Please resave the image as JPEG, PNG, or WebP."
                    ),
                )
            pixels = width * height
            if (
                width <= 0
                or height <= 0
                or width > MAX_IMAGE_WIDTH
                or height > MAX_IMAGE_HEIGHT
                or pixels > MAX_IMAGE_PIXELS
            ):
                return EnrollmentImage(
                    source_file,
                    error=(
                        "Image dimensions exceed safe pixel limit. "
                        f"Please select an image at most {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT} and "
                        f"{MAX_IMAGE_PIXELS:,} pixels."
                    ),
                )
            image.verify()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return EnrollmentImage(
            source_file,
            error="Image header or file data is corrupted. Please select another image.",
        )

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return EnrollmentImage(
            source_file,
            error="Could not read image. Please select another JPEG, PNG, or WebP file.",
        )
    height, width = frame.shape[:2]
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT or width * height > MAX_IMAGE_PIXELS:
        return EnrollmentImage(
            source_file,
            error="Decoded image exceeds safe pixel limit. Please select a smaller image.",
        )
    return EnrollmentImage(source_file, image_bgr=frame)


def decode_uploaded_capture(
    raw: bytes,
    *,
    filename: str,
    content_type: str,
    index: int,
    source_file: str | None = None,
) -> EnrollmentImage:
    """Validate the real format and decode one multipart JPEG/PNG/WebP image."""
    safe_name = Path(filename or f"image-{index}").name
    source_file = source_file or f"upload/{index:03d}_{safe_name}"
    mime_type = (content_type or "").split(";", 1)[0].lower()
    suffix = Path(safe_name).suffix.lower()
    if mime_type not in UPLOAD_TYPES or suffix not in UPLOAD_TYPES[mime_type]:
        return EnrollmentImage(
            source_file,
            error="Image type is not supported. Please select a JPEG, PNG, or WebP file.",
        )
    return _decode_image_payload(
        raw,
        source_file=source_file,
        filename=safe_name,
        content_type=mime_type,
    )


def _quality_guidance(reason: str) -> str:
    if reason.startswith("small"):
        return "Face is too small. Move closer to the camera and center your face in the frame."
    if reason.startswith("aligner"):
        return "Alignment score is low. Look straight at the camera without covering your face."
    if reason.startswith("blur"):
        return "Image is blurry. Clean the lens, increase lighting, and hold still."
    if reason.startswith("yaw") or reason.startswith("pitch"):
        return "Face angle is too steep. Look directly into the camera and retake."
    return "Face quality is insufficient. Adjust lighting and head pose, then retake."


@dataclass
class EnrollReport:
    people: int = 0
    images_seen: int = 0
    embedded: int = 0
    no_face: list[str] = field(default_factory=list)
    no_align: list[str] = field(default_factory=list)
    outliers: list[tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.people} people, {self.embedded}/{self.images_seen} images embedded; "
                f"{len(self.no_face)} no-face, {len(self.no_align)} no-align, "
                f"{len(self.outliers)} outliers")


class Enroller:
    def __init__(self, detector=None, aligner=None, recognizer=None):
        self.detector = detector or build_detector(
            settings.detector_kind,
            settings.model_path(settings.detector_model),
            imgsz=settings.detect_width,
            conf=settings.detect_conf,
        )
        self.aligner = aligner or FaceAligner(
            settings.model_path(settings.aligner_model),
            crop_size=settings.align_crop_size,
            margin=settings.align_margin,
            mode=settings.align_mode,
        )
        self.recognizer = recognizer or FaceRecognizer(
            settings.model_path(settings.recognizer_model), batch_size=settings.embed_batch
        )

    # -- one image --------------------------------------------------------
    def embed_file(self, path: Path):
        """-> (embedding, aligner_score, sharpness, face_px) or None."""
        bgr = cv2.imread(str(path))
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        dets = self.detector.detect(bgr)
        if not dets:
            return None
        d = max(dets, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))

        faces = self.aligner.align(rgb, [d.box])
        if not faces:
            return None
        f = faces[0]

        emb = self.recognizer.embed(f.aligned[None])[0]
        face_px = float(max(d.box[2] - d.box[0], d.box[3] - d.box[1]))
        return emb, f.score, sharpness_of(f.aligned), face_px

    def embed_capture(self, image_bgr: np.ndarray) -> CaptureEmbedding:
        """Validate and embed one browser capture with the active pipeline gates."""
        dets = self.detector.detect(image_bgr)
        if not dets:
            raise EnrollmentCaptureError(
                "No face detected. Center your face in the frame and retake."
            )
        detection = max(
            dets,
            key=lambda item: (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]),
        )
        faces = self.aligner.align(image_bgr, [detection.box], is_bgr=True)
        if not faces:
            raise EnrollmentCaptureError(
                "Could not align face. Look straight at the camera and retake."
            )
        face = faces[0]
        quality = assess(
            detection.box,
            face.aligned,
            face.landmarks,
            face.score,
            min_face_px=settings.min_face_px,
            min_laplacian_var=settings.min_laplacian_var,
            min_aligner_score=settings.min_aligner_score,
            max_yaw_deg=settings.max_yaw_deg,
            max_pitch_deg=settings.max_pitch_deg,
        )
        if not quality.ok:
            raise EnrollmentCaptureError(_quality_guidance(quality.reason))

        try:
            vector = self.recognizer.embed(face.aligned[None])[0]
        except Exception as exc:
            raise EnrollmentCaptureError(
                "Could not generate embedding. Please retake samples; if problem persists contact administrator."
            ) from exc
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            raise EnrollmentCaptureError(
                "Embedding result is invalid. Please retake samples; if problem persists contact administrator."
            )
        return CaptureEmbedding(vector=vector, quality=float(quality.aligner_score))

    # -- full gallery -----------------------------------------------------
    def run(self, gallery_dir: Path | None = None, outlier_threshold: float = 0.35,
            wipe: bool = True) -> EnrollReport:
        gallery_dir = Path(gallery_dir or settings.gallery_dir)
        rep = EnrollReport()
        model_name = settings.recognizer_model

        folders = sorted(
            p for p in gallery_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )

        # Everything expensive happens first, with NO transaction open. This
        # used to run inside the write transaction below, so the SQLite write
        # lock was held for the whole decode/detect/align/embed of every image
        # - ten seconds and more - while the capture threads sat blocked on it.
        prepared: list[tuple[Path, dict, str, list]] = []
        for folder in folders:
            meta_path = folder / "metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            ext_id = meta.get("user_id") or folder.name.split("_", 1)[-1]

            rows, vecs = [], []
            for img in sorted(folder.glob("*.png")):
                rep.images_seen += 1
                out = self.embed_file(img)
                if out is None:
                    rep.no_face.append(f"{folder.name}/{img.name}")
                    continue
                emb, ascore, sharp, face_px = out
                rows.append((img.name, emb, ascore))
                vecs.append(emb)

            # Flag images that disagree with the rest of their own folder.
            if len(vecs) >= 3:
                V = np.stack(vecs)
                S = V @ V.T
                np.fill_diagonal(S, np.nan)
                cohesion = np.nanmean(S, axis=1)
                for (name, _e, _q), c in zip(rows, cohesion):
                    if c < outlier_threshold:
                        rep.outliers.append((f"{folder.name}/{name}", float(c)))
            prepared.append((folder, meta, ext_id, rows))

        # One short transaction: the wipe and every upsert together.
        with session_scope() as s:
            if wipe:
                # Enrolment rows only. A `live:` row is an admin-reviewed
                # corridor crop carrying its own floor (app/services/augment.py);
                # re-enrolling used to throw those away along with the
                # photographs they were added to supplement.
                s.execute(delete(FaceEmbedding).where(or_(
                    FaceEmbedding.source_file.is_(None),
                    FaceEmbedding.source_file.not_like(f"{TAG}%"))))
                # ...except live rows built by a DIFFERENT recognizer. Those are
                # meaningless to this one, and keeping them would make
                # load_gallery() refuse the very gallery this run rebuilds.
                stale = [i for i, m in s.execute(
                    select(FaceEmbedding.id, FaceEmbedding.model_name)
                    .where(FaceEmbedding.source_file.like(f"{TAG}%"))).all()
                    if _model_key(m) != _model_key(model_name)]
                if stale:
                    s.execute(delete(FaceEmbedding).where(FaceEmbedding.id.in_(stale)))
                    log.warning("dropped %d live row(s) built by another recognizer",
                                len(stale))
                s.flush()

            for folder, meta, ext_id, rows in prepared:
                emp = s.execute(
                    select(Employee).where(Employee.external_id == ext_id)
                ).scalar_one_or_none()
                if emp is None:
                    emp = Employee(external_id=ext_id)
                    s.add(emp)
                emp.folder = folder.name
                emp.full_name = meta.get("full_name") or ext_id
                emp.department = meta.get("department", "") or ""
                emp.position = meta.get("position", "") or ""
                emp.phone = meta.get("phone", "") or ""
                emp.user_type = meta.get("user_type", "") or ""
                emp.is_active = True
                s.flush()

                for name, emb, ascore in rows:
                    s.add(FaceEmbedding(
                        employee_id=emp.id, source_file=name,
                        vector=emb.astype(np.float32).tobytes(), dim=int(emb.shape[0]),
                        model_name=model_name, quality=float(ascore),
                    ))
                    rep.embedded += 1
                rep.people += 1
                log.info("enrolled %-28s %d images", folder.name, len(rows))

        return rep


def enroll_employee_captures(
    *,
    full_name: str,
    position: str,
    department: str,
    phone_number: str,
    captures: list[EnrollmentImage],
    enroller: Enroller,
) -> EnrollmentResult:
    """Create one employee transaction from the valid subset of all input images."""
    name = full_name.strip()
    if not name:
        raise EnrollmentCaptureError("Please enter employee's full name.")
    if not captures:
        raise EnrollmentCaptureError(
            "No face samples submitted. Please retake samples and submit again."
        )

    embedded: list[tuple[str, CaptureEmbedding]] = []
    rejected: list[EnrollmentRejection] = []
    for index, capture in enumerate(captures, start=1):
        if index > MAX_BROWSER_CAPTURES:
            rejected.append(EnrollmentRejection(
                source_file=capture.source_file,
                error=(
                    f"Sample {index} rejected: at most "
                    f"{MAX_BROWSER_CAPTURES} images can be submitted."
                ),
            ))
            continue
        if capture.error:
            rejected.append(EnrollmentRejection(
                source_file=capture.source_file,
                error=f"Sample {index} rejected: {capture.error}",
            ))
            continue
        try:
            result = enroller.embed_capture(capture.image_bgr)
        except EnrollmentCaptureError as exc:
            rejected.append(EnrollmentRejection(
                source_file=capture.source_file,
                error=f"Sample {index} rejected: {exc}",
            ))
            continue
        except Exception as exc:
            log.exception("capture extraction failed for %s", capture.source_file)
            rejected.append(EnrollmentRejection(
                source_file=capture.source_file,
                error=(
                    f"Sample {index} rejected: could not process sample. "
                    "Please retake; if problem persists contact administrator."
                ),
            ))
            continue

        vector = np.asarray(result.vector, dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            rejected.append(EnrollmentRejection(
                source_file=capture.source_file,
                error=f"Sample {index} rejected: invalid embedding result. Please retake.",
            ))
            continue
        embedded.append((
            capture.source_file,
            CaptureEmbedding(vector=vector, quality=float(result.quality)),
        ))

    if not embedded:
        raise EnrollmentCaptureError(
            "None of the face samples produced a valid embedding. Fix the errors below and try again.",
            rejected,
        )

    external_id = f"WEB-{uuid.uuid4().hex[:12].upper()}"
    with session_scope() as session:
        employee = Employee(
            external_id=external_id,
            full_name=name,
            department=department.strip(),
            position=position.strip(),
            phone=phone_number.strip(),
            is_active=True,
        )
        session.add(employee)
        session.flush()
        session.add_all([
            FaceEmbedding(
                employee_id=employee.id,
                source_file=source_file,
                vector=item.vector.tobytes(),
                dim=int(item.vector.shape[0]),
                model_name=settings.recognizer_model,
                quality=item.quality,
            )
            for source_file, item in embedded
        ])
        session.flush()
        result = EnrollmentResult(
            employee_id=int(employee.id),
            external_id=external_id,
            embeddings=len(embedded),
            rejected=tuple(rejected),
        )
    return result


def _model_key(name) -> str:
    """Identity of a recognizer for gallery-compatibility purposes.

    Two things are deliberately NOT part of it:

    * `.enc` - the same weights, encrypted for a licensed deployment.
    * the precision suffix - `_fp16` and `_fp32` of one export produce the same
      embeddings to well inside the noise floor (d-prime 10.0271 vs 10.0283 on
      this gallery), so a gallery built by one is valid for the other. The live
      gallery is stamped `adaface_ir101_finetune.onnx` while the deployed model
      is `adaface_ir101_finetune_fp16.onnx`; treating those as incompatible
      would refuse to start on a gallery that is perfectly good.

    What it MUST separate is genuinely different architectures, whose
    embeddings are not comparable at all.
    """
    n = Path(str(name or "")).name
    n = n.replace(".enc", "")
    for suffix in (".onnx", ".pt"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    for prec in ("_fp16", "_fp32", "_int8"):
        if n.endswith(prec):
            n = n[: -len(prec)]
    return n


def load_gallery(strict: bool = True) -> Gallery:
    """Read every embedding out of the database into one matrix.

    `strict` refuses a gallery that was not built by the recognizer now loaded.
    Embeddings from two different models are not comparable, but nothing about
    them says so: both are 512-d and L2-normalised, so the matmul succeeds and
    returns entirely plausible cosine scores against the wrong vectors. There is
    no error, no warning, and nothing in the data to reveal it - just silently
    wrong matching, in the gallery AND in every live comparison at once.

    That is exactly what switching `recognizer_model` without re-enrolling does,
    and it is why this is checked rather than documented.
    """
    with session_scope() as s:
        rows = s.execute(
            select(FaceEmbedding.employee_id, FaceEmbedding.vector,
                   Employee.full_name, Employee.is_active,
                   FaceEmbedding.model_name, FaceEmbedding.threshold)
            .join(Employee, Employee.id == FaceEmbedding.employee_id)
            .where(Employee.is_active.is_(True))
        ).all()

    if not rows:
        return Gallery(np.zeros((0, 512), np.float32), np.zeros((0,), np.int64), {})

    want = _model_key(settings.recognizer_model)
    got = {_model_key(r[4]) for r in rows}
    if strict and got != {want}:
        listed = ", ".join(sorted(x or "<unset>" for x in got))
        raise RuntimeError(
            f"gallery/recognizer mismatch: the loaded recognizer is {want!r} but "
            f"the {len(rows)} stored embeddings were built by {{{listed}}}. "
            f"Vectors from one recognizer are meaningless to another - the "
            f"comparison still succeeds and returns plausible-looking scores, "
            f"which is why this refuses rather than warns. "
            f"Run `python scripts/enroll.py` to rebuild the gallery.")

    vecs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    owners = np.array([r[0] for r in rows], dtype=np.int64)
    names = {int(r[0]): r[2] for r in rows}
    # 0.0 where a row has no floor of its own; Gallery reads that as "judge
    # this one against the global threshold", which is every enrolment photo.
    floors = np.array([float(r[5] or 0.0) for r in rows], dtype=np.float32)
    return Gallery(vecs, owners, names, floors)
