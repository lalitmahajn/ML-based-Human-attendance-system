#!/usr/bin/env python3
"""Process N frames from every enabled camera and report real timings."""
import logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sqlalchemy import select

from app.config import settings
from app.core.pipeline import CameraPipeline
from app.core.stream import RtspSource
from app.db.models import Camera
from app.db.session import session_scope
from app.services.enrollment import load_gallery

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

def main():
    gallery = load_gallery()
    print(f"gallery: {len(gallery)} embeddings / {gallery.n_people} people\n")

    with session_scope() as s:
        cams = s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars().all()
        cams = [(c.id, c.name, c.role.value, c.rtsp_url) for c in cams]

    for cid, name, role, url in cams:
        print("=" * 74)
        print(f"camera {cid}  {name}  role={role}")
        print("=" * 74)
        if settings.replay_dir:
            from app.core.stream import ReplaySource
            src = ReplaySource(Path(settings.replay_dir) / name, name=name,
                               fps=settings.track_frame_rate,
                               queue_size=settings.frame_queue_size,
                               stale_after_s=settings.stale_after_s).start()
        else:
            src = RtspSource(url, name=name, transport=settings.rtsp_transport,
                             queue_size=settings.frame_queue_size,
                             stale_after_s=settings.stale_after_s).start()
        from app.core.direction import config_from_camera
        with session_scope() as s:
            cam_obj = s.get(Camera, cid)
            dir_cfg = config_from_camera(cam_obj) if cam_obj else None
        pipe = CameraPipeline(name, gallery, direction_cfg=dir_cfg)

        t_wait = time.time()
        while not src.connected and time.time() - t_wait < 20:
            time.sleep(0.3)
        if not src.connected:
            print("  FAILED to connect\n"); src.stop(); continue

        stats, seen, nth = [], 0, 0
        detected_any = 0
        t0 = time.time()
        while seen < N and time.time() - t0 < 60:
            f = src.read(timeout=2.0)
            if f is None: continue
            nth += 1
            if nth % settings.process_every_nth: continue
            r = pipe.process(f)
            stats.append(r.timings); seen += 1
            if r.timings["faces"]: detected_any += 1
            for o in r.outcomes:
                print(f"  >>> RECOGNIZED  {o.name}  score={o.score:.3f} margin={o.margin:.3f} "
                      f"face={o.face_px}px votes={o.votes}")

        if stats:
            def col(k): return np.array([s[k] for s in stats], dtype=float)
            print(f"\n  {src.stats()}")
            print(f"  processed {len(stats)} frames, {detected_any} with a face detected")
            print(f"  {'stage':<10} {'mean':>8} {'p50':>8} {'p95':>8}  (ms)")
            for k in ("detect", "align", "embed", "total"):
                c = col(k)
                print(f"  {k:<10} {c.mean():8.1f} {np.percentile(c,50):8.1f} {np.percentile(c,95):8.1f}")
            tot = col("total")
            print(f"\n  sustained capacity: {1000/tot.mean():.1f} fps/camera "
                  f"(need {12/settings.process_every_nth:.0f} fps)")
        src.stop()
        print()

if __name__ == "__main__":
    main()
