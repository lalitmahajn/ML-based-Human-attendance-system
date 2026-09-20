#!/usr/bin/env python3
"""Configure each camera's direction tripwire.

Direction of travel is what decides check-in vs check-out — not which camera saw
the face.  Each camera needs a line across the walkway and a declaration of
which side is "inside".

Three ways to use it:

    python scripts/set_direction.py --preview
        Render every camera's current view with its line drawn on it, into
        data/direction/.  Look at the images, then set the line.

    python scripts/set_direction.py --camera 1 --line 0.0,0.45,1.0,0.55 \
                                    --inside below --depth grow
        Set the line for one camera.  Coordinates are normalized 0..1,
        x1,y1,x2,y2.  --inside says which side of the line is inside the
        building: above | below | left | right.  --depth says whether a face
        GROWS (grow) or SHRINKS (shrink) as the person walks inward.

    python scripts/set_direction.py --show
        Print what is currently configured.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from sqlalchemy import select

from app.config import settings
from app.core.direction import DirectionConfig, _side, config_from_camera
from app.core.stream import RtspSource
from app.db.models import Camera
from app.db.session import session_scope

OUT = settings.data_dir / "direction"


def grab(url: str, name: str):
    if settings.replay_dir:
        from app.core.stream import ReplaySource
        r = ReplaySource(Path(settings.replay_dir) / name, name=name)
        clips = r.clips()
        if clips:
            cap = cv2.VideoCapture(str(clips[0]))
            # Grab a frame ~1 second in so it's not a black initial frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return frame

    src = RtspSource(url, name=name, transport="tcp", queue_size=2).start()
    import time
    t0 = time.time()
    while time.time() - t0 < 20:
        f = src.read(timeout=2.0)
        if f is not None:
            src.stop()
            return f.image
    src.stop()
    return None



def draw(img, cfg: DirectionConfig, label: str):
    h, w = img.shape[:2]
    out = cv2.resize(img, (1280, int(1280 * h / w)))
    H, W = out.shape[:2]
    if cfg.line:
        (x1, y1), (x2, y2) = cfg.line
        p1 = (int(x1 * W), int(y1 * H))
        p2 = (int(x2 * W), int(y2 * H))
        cv2.line(out, p1, p2, (0, 230, 255), 3)

        # Shade the inside half so the orientation is unmistakable.
        overlay = out.copy()
        ys, xs = np.mgrid[0:H, 0:W]
        sgn = ((x2 - x1) * (ys / H - y1) - (y2 - y1) * (xs / W - x1))
        mask = (np.sign(sgn) == cfg.inside_side)
        overlay[mask] = (0, 160, 0)
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.line(out, p1, p2, (0, 230, 255), 3)

        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(out, "INSIDE (green)", (mid[0] - 90, mid[1] + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, "INSIDE (green)", (mid[0] - 90, mid[1] + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 255, 60), 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "NO LINE CONFIGURED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, "NO LINE CONFIGURED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (60, 60, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (16, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (16, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def inside_sign(line, keyword: str) -> int:
    """Translate 'above/below/left/right' into the cross-product sign."""
    (x1, y1), (x2, y2) = line
    probes = {"above": (0.5, 0.0), "below": (0.5, 1.0),
              "left": (0.0, 0.5), "right": (1.0, 0.5)}
    if keyword not in probes:
        raise SystemExit(f"--inside must be one of {', '.join(probes)}")
    px, py = probes[keyword]
    s = _side(line, px, py)
    if abs(s) < 1e-9:
        raise SystemExit("that reference point lies on the line; pick another --inside")
    return 1 if s > 0 else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--line", help="x1,y1,x2,y2 normalized 0..1")
    ap.add_argument("--inside", choices=["above", "below", "left", "right"])
    ap.add_argument("--depth", choices=["grow", "shrink"], default="grow",
                    help="does the face GROW or SHRINK as the person walks inward")
    # No default: with one, every --line re-run silently reset a min_travel
    # that had been tuned on site. Unset, the stored value is left alone.
    ap.add_argument("--min-travel", type=float, default=None,
                    help="how far a track must move to count as walking "
                         "(normalized; the camera keeps its current value if omitted)")
    a = ap.parse_args()

    if a.camera and a.line:
        pts = [float(v) for v in a.line.split(",")]
        if len(pts) != 4:
            raise SystemExit("--line needs x1,y1,x2,y2")
        line = ((pts[0], pts[1]), (pts[2], pts[3]))
        if not a.inside:
            raise SystemExit("--inside is required with --line")
        sign = inside_sign(line, a.inside)
        with session_scope() as s:
            cam = s.get(Camera, a.camera)
            if not cam:
                raise SystemExit(f"no camera {a.camera}")
            cam.line_x1, cam.line_y1, cam.line_x2, cam.line_y2 = pts
            cam.inside_side = sign
            cam.depth_grows_inward = (a.depth == "grow")
            if a.min_travel is not None:
                cam.min_travel = a.min_travel
            print(f"  camera {cam.id} ({cam.name}): line={pts} inside={a.inside} "
                  f"(sign {sign:+d}) depth={a.depth} min_travel={cam.min_travel}")
        print("saved - restart the server to apply")
        return

    with session_scope() as s:
        cams = [(c.id, c.name, c.role.value, c.rtsp_url, config_from_camera(c))
                for c in s.execute(select(Camera)).scalars()]

    if a.show or not a.preview:
        print(f"{'id':>3}  {'camera':<10} {'role':<5} {'line (x1,y1,x2,y2)':<28} {'inside':>7} {'depth':>7}")
        print("-" * 72)
        for cid, name, role, _u, cfg in cams:
            ln = (f"{cfg.line[0][0]:.2f},{cfg.line[0][1]:.2f},"
                  f"{cfg.line[1][0]:.2f},{cfg.line[1][1]:.2f}") if cfg.line else "-- NOT SET --"
            print(f"{cid:>3}  {name:<10} {role:<5} {ln:<28} {cfg.inside_side:>+7} "
                  f"{'grow' if cfg.depth_grows_inward else 'shrink':>7}")
        if not a.preview:
            print("\nNothing decides check-in vs check-out correctly until every camera has a line.")
            print("Run with --preview to see the current views, then --camera N --line ... --inside ...")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for cid, name, role, url, cfg in cams:
        img = grab(url, name)
        if img is None:
            print(f"  {name}: could not grab a frame"); continue
        dst = OUT / f"camera{cid}_{name}.png"
        cv2.imwrite(str(dst), draw(img, cfg, f"camera {cid} · {name} · role {role}"))
        print(f"  {name}: {dst}")
    print(f"\nlook at {OUT}/ then set the lines")


if __name__ == "__main__":
    main()
