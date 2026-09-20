"""RTSP frame source.

Three behaviours the previous reader got wrong, all of which produce wrong
attendance data rather than a visible crash:

* **A dead camera must not replay its last frame.**  The old loop substituted
  `last_good_frame` on read failure and pushed it into the processing queue, so
  a camera that had been offline for hours kept generating recognitions from the
  face frozen in that frame.  Here a stale source stops producing entirely and
  reports `is_stale`.
* **Reconnect must use the same construction path.**  The old code opened with
  `CAP_FFMPEG` plus options and reconnected with a bare `VideoCapture(source)`.
* **The queue is drop-oldest.**  Recognition wants the newest frame; a backlog
  is latency, not work.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    image: np.ndarray      # BGR, full resolution
    ts: float              # wall-clock capture time (time.time())
    index: int


class RtspSource:
    """Background reader thread with bounded drop-oldest queue and backoff."""

    BACKOFF = (1, 2, 4, 8, 15, 30)

    def __init__(
        self,
        url: str,
        name: str = "cam",
        transport: str = "tcp",
        queue_size: int = 2,
        stale_after_s: float = 5.0,
        open_timeout_ms: int = 8000,
    ):
        self.url = url
        self.name = name
        self.transport = transport
        self.stale_after_s = stale_after_s
        self.open_timeout_ms = open_timeout_ms

        self.q: queue.Queue[Frame] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.connected = False
        self.last_frame_ts = 0.0
        self.frames_read = 0
        self.reconnects = 0
        self.read_failures = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "RtspSource":
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            # A read blocked on a silent camera returns after the read
            # timeout; give the thread that long to notice the stop.
            self._thread.join(timeout=self.stale_after_s + 2.0)

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.last_frame_ts) > self.stale_after_s

    # -- internals --------------------------------------------------------
    def _open(self) -> cv2.VideoCapture | None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.transport}|fflags;nobuffer|flags;low_delay"
        )
        # Timeouts as capture properties, which OpenCV enforces itself through
        # its interrupt callback whatever FFmpeg sits underneath. They used to
        # be passed as the `stimeout` option, which FFmpeg 5 removed; the
        # bundled build ignored it silently and the defaults applied - 30 s
        # to open and 30 s per read. A camera that stalled without closing
        # the socket (a reboot, a switch flap, a half-open connection) then
        # blocked every read for 30 s, and the thirty-failure rule below took
        # a quarter of an hour to notice, with /api/health saying "stale" the
        # whole time and nothing acting on it. Measured against a black-holed
        # host: open failed after 3.0 s with the property, 30.1 s without.
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout_ms),
                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(max(1.0, self.stale_after_s) * 1000)]
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _put(self, frame: Frame):
        """Drop-oldest: latency matters more than completeness."""
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self):
        """Reader loop. Any exception reconnects; none of them may end it.

        This had no exception handler at all. A cv2.error out of `read()` or
        `_open()` killed the thread outright, and nothing restarted it: the
        capture was never released (an fd leak), `connected` stayed True, and
        the only outward sign was `is_stale` going true - which /api/health
        reports but no supervisor acts on. The camera was simply gone until
        somebody restarted the service.
        """
        while not self._stop.is_set():
            try:
                self._read_loop()
            except Exception:
                self.connected = False
                log.exception("[%s] reader crashed; reconnecting in 2s", self.name)
                self._stop.wait(2.0)

    def _read_loop(self):
        attempt = 0
        cap = None
        opened_at = 0.0
        t_fps = time.time()
        n_fps = 0

        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = self._open()
                    if cap is None:
                        delay = self.BACKOFF[min(attempt, len(self.BACKOFF) - 1)]
                        attempt += 1
                        self.connected = False
                        log.warning("[%s] open failed, retry in %ds (attempt %d)",
                                    self.name, delay, attempt)
                        self._stop.wait(delay)
                        continue
                    attempt = 0
                    opened_at = time.time()
                    self.connected = True
                    self.reconnects += 1
                    log.info("[%s] connected %dx%d", self.name, self.width, self.height)

                ok, img = cap.read()
                if not ok or img is None:
                    self.read_failures += 1
                    # A few dropped frames are normal. A link is dead when
                    # nothing has arrived for `stale_after_s` - each failed
                    # read now takes at most that long, so this is one or two
                    # reads, not a run of thirty at 30 s apiece.
                    silent = time.time() - max(self.last_frame_ts, opened_at)
                    if self.read_failures >= 30 or silent > self.stale_after_s:
                        log.warning("[%s] stream lost (%.0fs without a frame), reconnecting",
                                    self.name, silent)
                        cap.release()
                        cap = None
                        self.connected = False
                        self.read_failures = 0
                        self._stop.wait(1.0)
                    continue

                self.read_failures = 0
                self.frames_read += 1
                self.last_frame_ts = time.time()
                self._put(Frame(img, self.last_frame_ts, self.frames_read))

                n_fps += 1
                if self.last_frame_ts - t_fps >= 2.0:
                    self.fps = n_fps / (self.last_frame_ts - t_fps)
                    n_fps = 0
                    t_fps = self.last_frame_ts
        finally:
            # Always released, including on the exception path - a leaked
            # VideoCapture holds its socket and the camera's stream slot.
            if cap is not None:
                cap.release()
            self.connected = False
        log.info("[%s] reader stopped", self.name)

    def read(self, timeout: float = 1.0) -> Frame | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> dict:
        return {
            "name": self.name, "connected": self.connected, "stale": self.is_stale,
            "fps": round(self.fps, 1), "frames": self.frames_read,
            "reconnects": self.reconnects, "resolution": f"{self.width}x{self.height}",
        }


class ReplaySource:
    """Recorded clips presented as a live stream.

    Same surface as `RtspSource` - `start/stop/read/stats`, `connected`,
    `is_stale`, `fps`, `width` - so `CameraWorker` cannot tell the difference.

    This exists because the cameras are reachable only from the office LAN,
    while the pipeline needs exercising from anywhere. Feeding it recorded
    footage is not a simulation: `scripts/record_4k.py` stream-copies the
    camera's own H.264/H.265 bitstream at native 3840x2160, so every frame the
    pipeline sees is exactly the frame the camera sent. That distinction
    matters - the earlier `data/recordings/` clips were re-encoded at
    `record_width=1920`, which halved every face and made the size gate look
    like the binding constraint when at true 4K it rejects 4%.

    Three details make the replay behave like a corridor rather than a video
    file, and each was a bug first:

    * **Real-time pacing.** Direction, the track buffer and the 90-point
      trajectory window are all defined in seconds. Replaying flat out would
      resolve direction from a trajectory spanning a fraction of its intended
      time.
    * **Original relative timing across cameras.** One walk was recorded by
      BOTH cameras, 1-4 s apart. Playing each camera's clips back-to-back turns
      those pairs into unrelated events minutes apart, so the cross-camera
      fusion never fires and the replay reports everything is fine because the
      situation that breaks it never arises.
    * **Only idle time is compressed.** Compressing on clip STARTS instead made
      ~16 s clips overlap at an 8 s spacing, so one camera played them
      back-to-back with no gap and tracks bridged across clips into a single
      55-second "pass" built from two different walks.
    """

    def __init__(self, clip_dir, name: str = "replay", fps: float = 20.0,
                 queue_size: int = 4, stale_after_s: float = 5.0,
                 loop: bool = True, max_gap_s: float | None = None):
        from pathlib import Path
        self.dir = Path(clip_dir)
        self.name = name
        self.target_fps = float(fps) if fps and fps > 0 else 20.0
        self.loop = loop
        self.max_gap_s = max_gap_s
        self.stale_after_s = stale_after_s

        self.q: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._durations: dict = {}

        self.connected = False
        self.last_frame_ts = 0.0
        self.frames_read = 0
        self.reconnects = 0
        self.read_failures = 0
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.clip = ""
        self.clips_played = 0

    # -- discovery --------------------------------------------------------
    def clips(self) -> list:
        return sorted(p for p in self.dir.glob("*.mp4")) if self.dir.is_dir() else []

    @staticmethod
    def _recorded_at(path):
        """Wall-clock start of a clip, from its `%Y%m%d_%H%M%S_<camera>` name."""
        from datetime import datetime
        try:
            return datetime.strptime("_".join(path.stem.split("_")[:2]),
                                     "%Y%m%d_%H%M%S").timestamp()
        except (ValueError, IndexError):
            return None

    def _duration(self, path) -> float:
        """Clip length in seconds, from its own metadata. Cached: the schedule
        is recomputed every loop and this opens the file."""
        key = str(path)
        if key not in self._durations:
            cap = cv2.VideoCapture(key)
            try:
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                f = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
                self._durations[key] = float(n) / f if n and f else 0.0
            finally:
                cap.release()
        return self._durations[key]

    def schedule(self, max_gap_s: float) -> list:
        """(offset_seconds, path) for this camera's clips, on a shared clock.

        Offsets come from the ORIGINAL recording times taken across every
        camera under the replay root, so both sources agree without talking to
        each other. Only genuinely IDLE stretches - where no camera is
        recording at all - are compressed; overlaps and short gaps, which is
        every cross-camera pair, are preserved exactly.
        """
        root = self.dir.parent
        every = sorted(root.glob("*/*.mp4")) if root.is_dir() else self.clips()
        stamped = [(self._recorded_at(p), p) for p in every]
        stamped = [(t, p) for t, p in stamped if t is not None]

        if not stamped:
            # Unparseable names: a plain sequence with a settling gap.
            out, run = [], 0.0
            for p in self.clips():
                out.append((run, p))
                run += self._duration(p) + max_gap_s
            return out

        stamped.sort()
        base = stamped[0][0]
        offsets, shift, busy_until = {}, 0.0, base
        for t, p in stamped:
            idle = t - busy_until
            if idle > max_gap_s:
                shift += idle - max_gap_s          # compress only the dead time
            offsets[p] = (t - base) - shift
            busy_until = max(busy_until, t + self._duration(p))
        return [(offsets.get(p, 0.0), p) for p in self.clips()]

    @property
    def is_stale(self) -> bool:
        return (self.last_frame_ts > 0
                and (time.time() - self.last_frame_ts) > self.stale_after_s)

    # -- lifecycle --------------------------------------------------------
    def start(self):
        found = self.clips()
        if not found:
            log.error("[%s] replay: no clips under %s", self.name, self.dir)
        else:
            log.info("[%s] replay: %d clip(s) from %s at %.0f fps",
                     self.name, len(found), self.dir, self.target_fps)
        self._thread = threading.Thread(target=self._run, name=f"replay-{self.name}",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.connected = False

    def read(self, timeout: float = 1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _put(self, frame: Frame):
        # Drop-oldest, exactly as the live source does: blocking here would
        # drift away from wall-clock pacing whenever the pipeline is busy.
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.q.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    # -- main -------------------------------------------------------------
    def _run(self):
        from app.config import settings
        max_gap = (self.max_gap_s if self.max_gap_s is not None
                   else settings.replay_max_gap_s)
        period = 1.0 / self.target_fps
        t_fps, n_fps = time.time(), 0

        while not self._stop.is_set():
            plan = sorted(self.schedule(max_gap))
            if not plan:
                self.connected = False
                if self._stop.wait(5.0):
                    break
                continue

            # A shared, deterministic epoch so both cameras start the same loop
            # at the same instant and keep the timing they were recorded with.
            # Quantised to five seconds, not to the whole span: both workers
            # start within milliseconds of each other and land in the same
            # slot either way, but rounding up to the span made the first
            # frame wait up to the length of the entire schedule.
            span = max(o + self._duration(p) for o, p in plan) + 10.0
            epoch = (int(time.time() / 5.0) + 1) * 5.0

            for offset, path in plan:
                if self._stop.is_set():
                    break
                due = epoch + offset
                wait = due - time.time()
                if wait > 0 and self._stop.wait(wait):
                    break
                if time.time() - due > 30.0:
                    continue                      # far behind: skip, do not pile up

                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    log.warning("[%s] replay: cannot open %s", self.name, path.name)
                    continue
                self.connected = True
                self.reconnects += 1
                self.clip = path.name
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                clip_fps = cap.get(cv2.CAP_PROP_FPS)
                period = 1.0 / clip_fps if (clip_fps and clip_fps > 0) else (1.0 / self.target_fps)
                next_due = time.time()
                try:
                    while not self._stop.is_set():
                        ok, img = cap.read()
                        if not ok:
                            break
                        # Paced on the wall clock; sleeping a fixed period would
                        # accumulate the decode time as drift.
                        next_due += period
                        delay = next_due - time.time()
                        if delay > 0:
                            if self._stop.wait(delay):
                                break
                        else:
                            next_due = time.time()
                        self.frames_read += 1
                        self.last_frame_ts = time.time()
                        self._put(Frame(img, self.last_frame_ts, self.frames_read))
                        n_fps += 1
                        if self.last_frame_ts - t_fps >= 2.0:
                            self.fps = n_fps / (self.last_frame_ts - t_fps)
                            n_fps, t_fps = 0, self.last_frame_ts
                finally:
                    cap.release()
                self.clips_played += 1

            if not self.loop:
                break
            # Long enough for every track to age out and be decided, rather
            # than merging into the first pass of the next loop.
            self.connected = False
            if self._stop.wait(settings.track_max_age_s + 2.0):
                break

        self.connected = False
        log.info("[%s] replay stopped after %d clip(s)", self.name, self.clips_played)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "connected": self.connected,
            "stale": self.is_stale,
            "fps": round(self.fps, 1),
            "frames": self.frames_read,
            "reconnects": self.reconnects,
            "resolution": f"{self.width}x{self.height}",
            "replay": {"dir": str(self.dir), "clip": self.clip,
                       "clips": len(self.clips()), "played": self.clips_played},
        }
