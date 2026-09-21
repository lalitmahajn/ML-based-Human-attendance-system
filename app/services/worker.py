"""Camera worker: joins the CV pipeline to the durable attendance state.

Runs one camera end to end — stream, recognition, event persistence — and keeps
a small snapshot of live state for the UI.  Designed to be run either in a
thread (one per camera) or as its own process; nothing here is shared between
cameras except the read-only gallery matrix.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.core.gallery import Gallery
from app.core.geometry import aligned_to_uint8
from app.core.direction import DirectionConfig
from app.core.pipeline import CameraPipeline, FrameResult
from app.core.stream import ReplaySource, RtspSource
from app.db.models import CameraRole, UnknownSighting
from app.db.session import session_scope
from app.services.attendance import AttendanceService
from app.services.debug_capture import DebugCapture
from app.services.recorder import ClipRecorder

log = logging.getLogger(__name__)


class CameraWorker:
    def __init__(self, camera_id: int, name: str, role: CameraRole, rtsp_url: str,
                 gallery: Gallery, direction_cfg: DirectionConfig | None = None,
                 arbiter=None, reid=None):
        self.camera_id = camera_id
        self.name = name
        self.role = role
        self.rtsp_url = rtsp_url
        self.direction_cfg = direction_cfg or DirectionConfig()

        if settings.replay_dir:
            # Recorded clips standing in for the camera. Same interface, so
            # nothing below this line knows the difference.
            self.source = ReplaySource(
                Path(settings.replay_dir) / name, name=name,
                fps=settings.track_frame_rate,
                queue_size=settings.frame_queue_size,
                stale_after_s=settings.stale_after_s,
            )
        else:
            self.source = RtspSource(
                rtsp_url, name=name, transport=settings.rtsp_transport,
                queue_size=settings.frame_queue_size, stale_after_s=settings.stale_after_s,
            )
        self.pipeline = CameraPipeline(name, gallery, direction_cfg=self.direction_cfg)
        self.attendance = AttendanceService()
        # Shared with the other camera: one walk past this corridor is seen by
        # both, and only one of them may move attendance state.
        from app.services.arbiter import PassArbiter
        self.arbiter = arbiter if arbiter is not None else PassArbiter()
        # Body-crop collection and cross-camera ReID. Everything it does happens
        # on its own thread; submitting never blocks this one.
        self.reid = reid
        self.debug = DebugCapture()
        self.recorder = (ClipRecorder(
            name, fps=max(1.0, 20.0 / settings.process_every_nth),
            pre_roll_s=settings.record_pre_roll_s,
            post_roll_s=settings.record_post_roll_s,
            width=settings.record_width,
            min_free_gb=settings.record_min_free_gb,
            max_clip_s=settings.record_max_clip_s,
        ) if settings.record_clips else None)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # live state for the UI
        self.latest: FrameResult | None = None
        self.latest_jpeg: bytes | None = None
        self._lock = threading.Lock()
        self.recent_events: list[dict] = []
        self.frames_seen = 0
        self.algorithm_fps: float = 0.0
        # Timestamp of the previous frame, to notice a break in the stream.
        self._last_frame_ts = 0.0
        # Person-pass accounting. A completed track is roughly one pass, and is
        # the denominator for "how many of the people who walked by did we
        # recognize" - a question recognition events alone cannot answer.
        self.passes_total = 0
        self.passes_recognized = 0
        self.passes_unknown = 0
        # Of `passes_recognized`, how many the live vote missed and the
        # whole-pass template recovered. Exposed because it is the one number
        # that says whether the second chance is earning its place: if it is
        # zero the rule is inert, and if it approaches the live count something
        # is wrong with the live gates rather than right with this.
        self.passes_recovered = 0
        self.passes_by_direction = {"ENTER": 0, "EXIT": 0, "UNKNOWN": 0}
        # A crash inside the per-frame path is silent from the outside: frames
        # keep flowing, timings look fine, and recognition simply stops. One
        # such bug ran 229 times before anyone noticed, so the count is exposed
        # on /api/health where it can be seen without reading the log.
        self.pipeline_errors = 0
        self.last_error = ""

        self.snapshot_dir = settings.media_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle --------------------------------------------------------
    def start(self):
        self.source.start()
        self._thread = threading.Thread(target=self._run, name=f"worker-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self.recorder is not None:
            self.recorder.stop()
        self.source.stop()
        if self._thread:
            self._thread.join(timeout=5.0)
        # Anybody still in view when the service stops has a live track with a
        # perfectly good identity that nothing would ever write. Finalize them,
        # then drain whatever the arbiter is still holding.
        try:
            completed = self.pipeline.flush()
            if completed:
                log.info("[%s] flushing %d in-flight track(s) at shutdown",
                         self.name, len(completed))
                self._persist_completed(FrameResult(frame=self.latest.frame
                                                    if self.latest else None,
                                                    completed=completed))
            groups = self.arbiter.drain()
            if groups:
                with session_scope() as s:
                    for g in groups:
                        self._apply(s, g.winner, winner=True)
                        for loser in g.others:
                            self._apply(s, loser, winner=False)
        except Exception:
            log.exception("[%s] shutdown flush failed", self.name)

    # -- helpers ----------------------------------------------------------
    def _scale_box(self, box, ctx_width: int):
        full_w = self.source.width or ctx_width
        r = ctx_width / float(full_w) if full_w else 1.0
        return [v * r for v in box]

    def _save_snapshot(self, crop: np.ndarray | None, employee_id: int, kind: str,
                       track_id: int = -1) -> str | None:
        if crop is None:
            return None
        try:
            # Camera and track in the name, not just a whole second. Two
            # cameras completing the same person's track in the same second is
            # routine here - it is the cross-camera pair the arbiter exists for
            # - and both wrote the same path, so one image silently replaced
            # the other and both database rows pointed at the survivor. Every
            # unknown was worse: they all pass employee_id=0.
            rel = (f"snapshots/{kind}_{employee_id}_{self.camera_id}_"
                   f"{track_id}_{int(time.time())}.jpg")
            rgb = aligned_to_uint8(crop)
            cv2.imwrite(str(settings.media_dir / rel), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            return rel
        except Exception:
            log.exception("[%s] snapshot write failed", self.name)
            return None

    def _save_body(self, crop, employee_id: int, track_id: int = -1) -> str | None:
        """Write a raw BGR body crop. Separate from _save_snapshot because that
        one expects an aligned CHW face in [-1, 1]; this is already an image."""
        if crop is None:
            return None
        try:
            rel = (f"snapshots/body_{employee_id}_{self.camera_id}_"
                   f"{track_id}_{int(time.time())}.jpg")
            cv2.imwrite(str(settings.media_dir / rel), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            return rel
        except Exception:
            log.exception("[%s] body snapshot write failed", self.name)
            return None

    def _second_chance(self, ct) -> tuple[int, float, float] | None:
        """One more look at a pass the live vote could not name.

        THE LIVE RULE THROWS AWAY EVIDENCE IT ALREADY HAS. Every frame is
        matched on its own and then `vote_min_recognitions` of them must agree,
        so a pass yielding three good frames names nobody however clearly each
        one scores. Those embeddings were computed on the capture thread and
        then discarded; combining them into one quality-weighted template and
        asking the SAME gallery once more costs a 512-wide matmul.

        MEASURED ON THIS RULE - not on the research prototype, which relaxed the
        live quality gates and therefore answered a different question. Running
        the real pipeline over 191 recorded clips of one day
        (`bench/tracklet_recheck_eval.py --stride 4 --sweep`, 66 passes):

            agreement where the live vote named somebody : 22/22 (100%)
            disagreements over thresholds 0.20 -> 0.40   : 0 everywhere
            recovered from the 38 unnamed passes         : 2  (+7%)

        Safe, and worth having. NOT the +23...30% the research reports for a
        gate-relaxed version of the same idea - see app/config.py for why the
        two numbers differ and what it would take to close the gap.

        WHY THE THRESHOLD EQUALS THE LIVE ONE RATHER THAN EXCEEDING IT. This is
        a second chance, not a lowered bar: `tracklet_face_threshold` is the
        live per-frame threshold exactly, so this can never accept a similarity
        the live matcher would reject. Everything it adds is the aggregation
        over the whole pass and the runner-up margin.

        It used to be 0.30, on the reasoning that a whole-pass template is a
        stronger query and could clear a higher bar. Production said otherwise:
        most of the enrolment gallery is WEBCAM photography, a CCTV query
        scores ~0.2 lower against those rows than against a CCTV one, and 0.30
        sat on the median of exactly the population this rule exists to
        recover. See app/config.py for the distributions.

        Returns (employee_id, score, margin), or None to leave the pass unknown.
        """
        thr = float(settings.tracklet_face_threshold)
        if thr <= 0.0 or ct.face_template is None:
            return None
        if ct.face_frames < settings.tracklet_min_face_frames:
            # One frame's template IS that frame, and the live matcher has
            # already judged it and said no. Re-asking the same question at a
            # different threshold is not new evidence.
            return None
        m = self.pipeline.gallery.match(ct.face_template, thr,
                                        settings.tracklet_face_margin)
        if m.employee_id is None:
            return None
        return int(m.employee_id), float(m.score), float(m.margin)

    def _persist_completed(self, res: FrameResult):
        """One attendance decision per PERSON-PASS - not per completed track.

        Identity and direction mature at different points in a walk: the vote
        commits within a few good frames, while direction needs real travel.
        Deciding at completion is the only moment both are final.

        Two things happen here that used to happen differently:

        * **Images are written before any transaction is opened.** Up to seven
          JPEG encodes plus a 24 MB full-frame copy ran while the SQLite write
          lock was held, because `_daily()` flushes and promotes the
          transaction. The other camera then blocked on `busy_timeout` waiting
          for this one's disk I/O, on the capture thread, inside the frame
          budget.
        * **Identified passes go to the arbiter, not straight to attendance.**
          Both cameras see the same corridor, so one walk arrives twice. The
          arbiter holds a pass briefly and lets only the strongest of a group
          move attendance state.
        """
        from app.services.attendance import business_date
        from app.services.arbiter import PendingPass

        now_mono = time.monotonic()
        unknowns = []
        # At shutdown there may be no frame to take a timestamp from.
        frame_ts = res.frame.ts if res.frame is not None else time.time()

        # ---- phase 1: all disk I/O, no transaction open --------------------
        for ct in (res.completed or []):
            self.passes_total += 1
            self.passes_by_direction[ct.direction] = \
                self.passes_by_direction.get(ct.direction, 0) + 1
            ts = datetime.fromtimestamp(frame_ts, tz=timezone.utc)

            source = "live"
            if ct.employee_id is None:
                # Before writing this off as an unknown: the whole pass as one
                # query, against the same gallery. See `_second_chance`.
                again = None
                try:
                    again = self._second_chance(ct)
                except Exception:
                    log.exception("[%s] tracklet re-match failed", self.name)
                if again is not None:
                    emp, score, margin = again
                    ct.employee_id, ct.name = emp, self.pipeline.gallery.name(emp)
                    ct.best_score, ct.best_margin = score, margin
                    source = "tracklet"
                    self.passes_recovered += 1
                    log.info("[%s] RECOVERED %-20s score=%.3f margin=%.3f "
                             "faces=%d ipd=%.0f dir=%s (the live vote saw %d "
                             "embedded frame(s) and named nobody)",
                             self.name, ct.name[:20], score, margin,
                             ct.face_frames, ct.best_ipd, ct.direction,
                             ct.embedded_frames)

            if ct.employee_id is None:
                self.passes_unknown += 1
                # The BODY is what the unknown-review page shows, exactly as it
                # is for a recognised pass. A 112x112 aligned face of somebody
                # the system could not place is the least useful image it
                # could offer a human: build, clothing and posture are what
                # make an unknown identifiable at all. The aligned face is
                # still written beside it for diagnosis.
                face_snap = self._save_snapshot(ct.crop, 0, "unknown", ct.track_id)
                snap = self._save_body(ct.person_crop, 0, ct.track_id) or face_snap
                unknowns.append((ct, ts, snap))
                if self.reid is not None:
                    self.reid.submit_pass(self.camera_id, self.name, ct, ts)
                log.info("[%s] MISS best=%.3f nearest=%-20s embedded=%d traj=%d "
                         "travel=%.3f dur=%.1fs dir=%s (%s)",
                         self.name, ct.best_score,
                         self.pipeline.gallery.name(ct.nearest_employee_id)[:20],
                         ct.embedded_frames, ct.traj_points, ct.travel,
                         ct.duration_s, ct.direction, ct.direction_reason)
                continue

            self.passes_recognized += 1
            # The dashboard shows the BODY: a person is recognisable to a human
            # by build, clothing and posture, where a 112x112 aligned face often
            # is not - especially on the crops that turn out to be wrong. The
            # aligned face is still written next to it under "evt_", so a
            # questionable match can still be examined.
            face_snap = self._save_snapshot(ct.crop, ct.employee_id, "evt", ct.track_id)
            snap = self._save_body(ct.person_crop, ct.employee_id, ct.track_id) or face_snap
            if ct.score_crop is not None:
                self._save_snapshot(ct.score_crop, ct.employee_id, "why", ct.track_id)
            if ct.context is not None and ct.box is not None:
                try:
                    self.debug.capture(
                        name=ct.name, frame_bgr=ct.context, box=self._scale_box(
                            ct.box, ct.context.shape[1]),
                        aligned_chw=ct.crop, camera=self.name, role=self.role.value,
                        score=ct.best_score, margin=ct.best_margin, track_id=ct.track_id,
                        ts=ts, quality=ct.quality, native=ct.native,
                        extra={"embedded_frames": ct.embedded_frames,
                               "traj_points": ct.traj_points,
                               "travel": round(ct.travel, 3),
                               "duration_s": round(ct.duration_s, 1),
                               "direction": ct.direction,
                               "direction_reason": ct.direction_reason,
                               "employee_id": ct.employee_id},
                    )
                except Exception:
                    log.exception("[%s] debug capture failed", self.name)

            if self.reid is not None:
                self.reid.submit_pass(self.camera_id, self.name, ct, ts)

            self.arbiter.submit(PendingPass(
                employee_id=ct.employee_id, camera_id=self.camera_id,
                role=self.role, ts=ts, monotonic=now_mono, track=ct,
                snapshot=snap, camera_name=self.name, source=source))

        # ---- phase 2: whatever is now decided, in one short transaction ----
        groups = self.arbiter.due(now_mono)
        if not unknowns and not groups:
            return
        with session_scope() as s:
            for ct, ts, snap in unknowns:
                s.add(UnknownSighting(
                    camera_id=self.camera_id, track_id=ct.track_id,
                    # first_seen was written equal to last_seen, so every
                    # duration in the table read 0.00s and the column could
                    # never answer "how long was this person in view" - the
                    # signal that separates a real transit from somebody
                    # standing still. duration_s is on the track; use it.
                    first_seen=ts - timedelta(seconds=ct.duration_s),
                    last_seen=ts, business_date=business_date(ts),
                    frames=ct.embedded_frames, best_score=ct.best_score,
                    nearest_employee_id=ct.nearest_employee_id,
                    vector=ct.vector.astype("float32").tobytes() if ct.vector is not None else None,
                    snapshot=snap,
                ))
            for g in groups:
                self._apply(s, g.winner, winner=True, direction=g.direction,
                            direction_reason=g.direction_reason)
                for loser in g.others:
                    self._apply(s, loser, winner=False)

    def _apply(self, s, p, *, winner: bool, direction: str | None = None,
               direction_reason: str | None = None):
        """Write one pass. Only the winner of a group moves attendance state,
        and it moves it in the direction the GROUP resolved - which may have
        come from the other camera's view of the same walk."""
        ct = p.track
        direction = ct.direction if direction is None else direction
        direction_reason = (ct.direction_reason if direction_reason is None
                            else direction_reason)
        d = self.attendance.record(
            s, employee_id=p.employee_id, camera_id=p.camera_id,
            role=p.role, ts=p.ts, score=ct.best_score, margin=ct.best_margin,
            track_id=ct.track_id, face_px=ct.face_px,
            votes=f"emb{ct.embedded_frames}/traj{ct.traj_points}",
            snapshot=p.snapshot, direction=direction,
            direction_reason=direction_reason,
            require_direction=self.direction_cfg.configured,
            apply_state=winner, source=p.source,
        )
        entry = {
            "ts": p.ts.astimezone(settings.tz).strftime("%H:%M:%S"),
            # The display string above cannot be sorted on - 23:59 outranks
            # 00:01 and the 04:00 business boundary interleaves two days.
            "sort_ts": p.ts.timestamp(),
            "name": ct.name, "employee_id": p.employee_id,
            "camera": p.camera_name or self.name, "role": p.role.value,
            "score": round(ct.best_score, 3), "transition": d.transition,
            "snapshot": p.snapshot, "direction": direction,
        }
        with self._lock:
            self.recent_events.insert(0, entry)
            del self.recent_events[40:]
        log.info("[%s] %-14s %-20s score=%.3f margin=%.3f embedded=%d traj=%d "
                 "travel=%.3f dur=%.1fs dir=%s (%s)",
                 p.camera_name or self.name, d.transition, ct.name[:20],
                 ct.best_score, ct.best_margin, ct.embedded_frames, ct.traj_points,
                 ct.travel, ct.duration_s, direction, direction_reason)

    # -- main loop --------------------------------------------------------
    def _run(self):
        log.info("[%s] worker started (role=%s)", self.name, self.role.value)
        nth = 0
        t_proc_fps = time.time()
        n_proc_fps = 0
        while not self._stop.is_set():
            frame = self.source.read(timeout=1.0)
            if frame is None:
                continue
            self.frames_seen += 1

            # A GAP IN THE STREAM IS A DISCONTINUITY, NOT A PAUSE.
            #
            # ByteTrack measures its lost-track buffer in FRAMES, not seconds
            # (max_time_lost = frame_rate/30 * track_buffer). When frames stop
            # arriving its clock stops with them, so after a stall it happily
            # re-associates tracks from before the gap with whoever is in view
            # afterwards - a different person inherits the earlier person's
            # track, and with it their vote and their identity.
            #
            # Seen in replay as a 55-second "pass" spanning two clips recorded
            # eight seconds apart. The same thing happens live whenever a
            # camera drops out and reconnects.
            #
            # So: finalize what was in view, and start the tracker clean.
            # The trigger is ByteTrack's OWN buffer expressed in real time
            # (track_buffer / track_frame_rate = 36/20 = 1.8 s): a gap longer
            # than that is precisely a gap in which it would have dropped the
            # track, had frames kept flowing. Anything shorter is jitter and
            # must not split one person's pass in two.
            gap_limit = max(1.0, settings.track_buffer / max(1, settings.track_frame_rate))
            if self._last_frame_ts and \
                    (frame.ts - self._last_frame_ts) > gap_limit:
                try:
                    stale = self.pipeline.flush()
                    if stale:
                        log.info("[%s] %.0fs gap in the stream: closing %d track(s)",
                                 self.name, frame.ts - self._last_frame_ts, len(stale))
                        self._persist_completed(FrameResult(frame=frame, completed=stale))
                    self.pipeline.tracker.reset()
                except Exception:
                    log.exception("[%s] failed to close tracks across a gap", self.name)
            self._last_frame_ts = frame.ts

            nth += 1
            if nth % settings.process_every_nth:
                continue
            try:
                res = self.pipeline.process(frame)
            except Exception as e:
                self.pipeline_errors += 1
                self.last_error = f"{type(e).__name__}: {e}"[:160]
                log.exception("[%s] pipeline error", self.name)
                continue
            with self._lock:
                self.latest = res

            n_proc_fps += 1
            now_proc = time.time()
            if now_proc - t_proc_fps >= 1.5:
                self.algorithm_fps = round(n_proc_fps / (now_proc - t_proc_fps), 1)
                n_proc_fps, t_proc_fps = 0, now_proc

            # Tell the arbiter who this camera still has in view, so a pass
            # waiting to be decided is not decided while its person's other
            # track is still running on either camera.
            self.arbiter.note_live(
                self.camera_id,
                {t.employee_id for t in res.tracks
                 if t.employee_id is not None and frame.ts - t.last_seen < 1.0},
                time.monotonic())

            # Hand the periodic body crops over and move on. The queue is
            # bounded and drop-oldest, so a slow or failed ReID cannot stall
            # capture - dropping a training sample is the right trade against
            # delaying recognition.
            if self.reid is not None and res.body_crops:
                self.reid.submit_crops(self.camera_id, self.name, res.body_crops)

            if self.recorder is not None:
                try:
                    live = [t for t in res.tracks if frame.ts - t.last_seen < 1.0]
                    self.recorder.offer(
                        frame.image, frame.ts, active=bool(live),
                        meta={"names": sorted({t.name for t in live if t.employee_id}),
                              "tracks": [t.track_id for t in live]})
                except Exception:
                    log.exception("[%s] recorder error", self.name)

            if res.candidates:
                ts_c = datetime.fromtimestamp(frame.ts, tz=timezone.utc)
                for c in res.candidates:
                    self.debug.frame(name=c.name, aligned_chw=c.aligned, native=c.native,
                                     score=c.score, camera=self.name, track_id=c.track_id,
                                     ts=ts_c, idx=c.index, quality=c.quality,
                                     accepted=c.accepted)
            try:
                self._persist_completed(res)
            except Exception:
                log.exception("[%s] completed-track persist error", self.name)
        log.info("[%s] worker stopped", self.name)

    # -- UI ---------------------------------------------------------------
    def render(self, max_width: int = 1280) -> bytes | None:
        """Annotated JPEG of the most recent processed frame."""
        with self._lock:
            res = self.latest
        if res is None:
            return None

        img = res.frame.image
        h, w = img.shape[:2]
        scale = min(1.0, max_width / w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img = img.copy()

        now = time.time()
        for t in res.tracks:
            if now - t.last_seen > 2.0:
                continue
            x1, y1, x2, y2 = (t.box * scale).astype(int)
            known = t.employee_id is not None
            color = (0, 200, 0) if known else (40, 120, 240)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = t.name if known else (t.last_reason or "…")
            if known and t.score:
                label += f" {t.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        algo_fps = getattr(self, "algorithm_fps", 0.0)
        banner = f"{self.name} [{self.role.value}]  cam:{self.source.fps:.0f}fps  algo:{algo_fps:.0f}fps  {res.timings['total']:.0f}ms"
        cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def stats(self) -> dict:
        with self._lock:
            res = self.latest
        return {
            "camera_id": self.camera_id, "name": self.name, "role": self.role.value,
            "stream": self.source.stats(),
            "algorithm_fps": getattr(self, "algorithm_fps", 0.0),
            "frames_processed": self.pipeline.frames_processed,
            "faces_embedded": self.pipeline.faces_embedded,
            "active_tracks": len(self.pipeline.tracks),
            "pipeline_errors": self.pipeline_errors,
            "recorder": self.recorder.stats() if self.recorder else None,
            "last_error": self.last_error,
            "passes": {
                "total": self.passes_total,
                "recognized": self.passes_recognized,
                "recovered": self.passes_recovered,
                "unknown": self.passes_unknown,
                "by_direction": dict(self.passes_by_direction),
            },
            "timings": res.timings if res else {},
        }
