"""WebSocket camera streaming, in the shape the original live page expects.

`templates/recognition/live.html` connects to `/ws/camera/<id>/` and draws
`{type:"frame", data:<base64 jpeg>}` messages onto a canvas.  This serves that
contract from the v3 workers.

Each connection renders its own frame from the worker's latest processed frame,
so opening a second viewer does not steal the first one's frames — the previous
implementation had every consumer pulling from one shared queue.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.security import COOKIE_NAME, read_session
from app.services import auth as auth_svc
from app.web.viewmodels import live_event

from app.config import settings
from app.runtime import runtime

log = logging.getLogger(__name__)
router = APIRouter()

FPS = settings.live_preview_fps

# One JPEG per processed frame, however many viewers are watching. Each
# connection used to call worker.render() for itself - a resize of a 4K frame
# plus an encode, tens of milliseconds - so N open tabs cost N encodes of the
# SAME frame, ten times a second. The cache is keyed on the identity of the
# worker's latest FrameResult plus its capture time (a recycled id alone could
# serve the previous image); `_render_busy` serialises the encode per camera
# so two viewers arriving together do not both do the work either.
_render_lock = threading.Lock()
_render_cache: dict[int, tuple[tuple, bytes | None]] = {}
_render_busy: dict[int, threading.Lock] = {}


def shared_jpeg(worker) -> bytes | None:
    """`worker.render()`, encoded at most once per frame across all viewers."""
    latest = getattr(worker, "latest", None)
    if latest is None:
        return None
    key = (id(latest), getattr(getattr(latest, "frame", None), "ts", None))
    cid = worker.camera_id
    with _render_lock:
        hit = _render_cache.get(cid)
        if hit is not None and hit[0] == key:
            return hit[1]
        busy = _render_busy.setdefault(cid, threading.Lock())
    with busy:
        with _render_lock:              # the viewer we waited on may have filled it
            hit = _render_cache.get(cid)
            if hit is not None and hit[0] == key:
                return hit[1]
        jpg = worker.render()
        with _render_lock:
            _render_cache[cid] = (key, jpg)
    return jpg


def frame_message(worker, camera_id: int) -> dict | None:
    """One tick of the camera socket: what to send, or None for nothing.

    A stale source - no frame for `stale_after_s` - used to be re-encoded and
    re-sent at 10 fps, so a camera that had been down for hours still looked
    live to anyone watching. It is now announced as stale WITHOUT a frame, and
    the page paints "Oqim eskirgan" over the last image it drew. A camera that
    has never delivered a frame at all says nothing while it connects: the
    page's own "Kadr kutilmoqda" is the right state for that.
    """
    source = worker.source
    if source.is_stale:
        if getattr(worker, "latest", None) is None:
            return None
        return {"type": "frame", "stale": True, "camera_id": camera_id,
                "fps": round(source.fps, 1)}
    jpg = shared_jpeg(worker)
    latest = getattr(worker, "latest", None)
    total_ms = (latest.timings or {}).get("total") if latest else None
    algo_fps = getattr(worker, "algorithm_fps", 0.0)
    if (not algo_fps or algo_fps <= 0) and total_ms:
        algo_fps = round(min(60.0, 1000.0 / max(1.0, float(total_ms))), 1)
    return {
        "type": "frame",
        "data": base64.b64encode(jpg).decode("ascii"),
        "camera_id": camera_id,
        "fps": round(source.fps, 1),
        "algo_fps": algo_fps,
        "delay_ms": round(float(total_ms), 1) if total_ms is not None else None,
        "stale": False,
    }


async def _pump(ws: WebSocket, camera_id: int):
    worker = runtime.workers.get(camera_id)
    if worker is None:
        await ws.close(code=4004, reason="camera not running")
        return
    try:
        while True:
            msg = await asyncio.to_thread(frame_message, worker, camera_id)
            if msg:
                await ws.send_json(msg)
            await asyncio.sleep(1 / FPS)
    except WebSocketDisconnect:
        pass
    except Exception as e:                       # client vanished mid-send
        log.debug("ws camera %s ended: %s", camera_id, e)


def _resolve(camera_id: str) -> int | None:
    if camera_id.isdigit():
        return int(camera_id)
    if camera_id in ("primary", "default"):
        ids = sorted(runtime.workers)
        return ids[0] if ids else None
    for w in runtime.workers.values():          # allow /ws/camera/entrance/
        if w.name.lower() == camera_id.lower() or w.role.value.lower() == camera_id.lower():
            return w.camera_id
    return None



async def _authed(ws: WebSocket, *, admin_only: bool = False) -> bool:
    """Reject an unauthenticated socket before accepting it.

    HTTP middleware does not run for WebSocket scopes, so without this check
    the live camera feed and the attendance push stay world-readable even
    though every page around them requires a login. Browsers send cookies on
    the WebSocket handshake, so the same signed session applies.

    1008 is "policy violation"; closing before accept() means no frame is ever
    sent to a client that has not signed in.
    """
    session = read_session(ws.cookies.get(COOKIE_NAME))
    if session is None:
        await ws.close(code=1008, reason="not authenticated")
        return False
    # HTTP middleware does not run for WebSocket scopes, so the admin-only
    # rule that keeps operators off /recognition and /video has to be
    # repeated here - otherwise the live camera feed stays reachable by
    # socket for exactly the accounts the page was hidden from.
    if admin_only and not auth_svc.can_admin(session):
        await ws.close(code=1008, reason="admin only")
        return False
    return True


@router.websocket("/ws/camera/{camera_id}/")
async def camera_ws(ws: WebSocket, camera_id: str):
    if not await _authed(ws, admin_only=True):
        return
    await ws.accept()
    cid = _resolve(camera_id)
    if cid is None:
        await ws.close(code=4004, reason="unknown camera")
        return
    await _pump(ws, cid)


@router.websocket("/ws/camera/{camera_id}")
async def camera_ws_noslash(ws: WebSocket, camera_id: str):
    await camera_ws(ws, camera_id)


@router.websocket("/ws/attendance/")
async def attendance_ws(ws: WebSocket):
    """Pushes recognition events to the live page as they happen."""
    if not await _authed(ws):
        return
    await ws.accept()
    seen: set[tuple] = set()
    try:
        while True:
            for ev in runtime.events(12):
                # The real timestamp and the person, not the "%H:%M:%S"
                # display string: two recognitions of one name within the
                # same second on one camera are two events, and the same
                # event seen on the next poll is one.
                key = (ev.get("sort_ts"), ev.get("employee_id"), ev.get("camera"))
                if key in seen:
                    continue
                seen.add(key)
                await ws.send_json({"type": "event", **live_event(ev)})
            if len(seen) > 400:
                seen = set(list(seen)[-200:])
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("attendance ws ended: %s", e)
