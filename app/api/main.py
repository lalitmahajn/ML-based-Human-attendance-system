"""FastAPI surface: dashboard, live views, attendance, health."""
from __future__ import annotations

import asyncio
import csv
import io
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from app.config import settings
from app.db.models import (
    Camera, DailyAttendance, Employee, PresenceStatus, RecognitionEvent,
)
from app.db.session import session_scope
from app.runtime import runtime
from app.services.attendance import business_date
from app.services.auth import ensure_default_admin
from app.api import auth as auth_router
from app.api import pages as pages_router
from app.api import ws as ws_router
from app.web.viewmodels import live_event

log = logging.getLogger(__name__)


def _sweep_person_crops() -> int:
    """Delete body-crop day folders older than `reid_retention_days`.

    These are images of people, including people the system never identified,
    so they get a real expiry rather than accumulating until a disk fills. The
    layout is deliberately date-first (`known/YYYYMMDD/...`) so ageing them out
    is a directory rename away from trivial.
    """
    import shutil
    from datetime import date as _date
    root = settings.persons_dir
    if not root.is_dir() or settings.reid_retention_days <= 0:
        return 0
    cutoff = _date.today() - timedelta(days=settings.reid_retention_days)
    removed = 0
    for kind in ("known", "unknown"):
        base = root / kind
        if not base.is_dir():
            continue
        for day in base.iterdir():
            if not day.is_dir() or len(day.name) != 8 or not day.name.isdigit():
                continue
            try:
                when = _date(int(day.name[:4]), int(day.name[4:6]), int(day.name[6:]))
            except ValueError:
                continue
            if when < cutoff:
                shutil.rmtree(day, ignore_errors=True)
                removed += 1
    return removed


async def _eod_sweep_loop():
    """Close out finished days from inside the service.

    `close_open_intervals` existed only in scripts/maintenance.py, driven by
    deploy/ematsy-maintenance.timer - which is not installed on either host, so
    it had never run. 21 rows still claimed people were inside the building on
    business dates up to five days old, and their final interval was never
    accumulated into worked_seconds.

    Attendance correctness may not depend on somebody remembering to install a
    cron job, so it runs here. `scripts/maintenance.py` still works for manual
    runs; the sweep is idempotent, so both running is harmless.
    """
    from datetime import timedelta
    from app.services.attendance import AttendanceService, business_date

    while True:
        now = datetime.now(settings.tz)
        target = now.replace(hour=settings.day_boundary_hour, minute=0,
                             second=0, microsecond=0) + timedelta(
                                 minutes=settings.eod_sweep_offset_min)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep(max(60.0, (target - now).total_seconds()))
        try:
            # Everything up to and including yesterday. Sweeping only one date
            # means a day the service was down is never revisited.
            yesterday = business_date(datetime.now(settings.tz)) - timedelta(days=1)
            def _sweep():
                with session_scope() as s:
                    return AttendanceService().close_open_intervals(s, yesterday)
            n = await asyncio.to_thread(_sweep)
            log.info("end-of-day sweep: flagged %d open interval(s) up to %s",
                     n, yesterday)
            # Body crops age out here rather than in a cron job, for the reason
            # close_open_intervals had to move in-process: the timer that was
            # supposed to run it had never been installed on either host.
            removed = await asyncio.to_thread(_sweep_person_crops)
            if removed:
                log.info("retention: removed %d day-folder(s) of body crops",
                         removed)
        except Exception:
            log.exception("end-of-day sweep failed")


def websocket_transport() -> str | None:
    """The library uvicorn will use for WebSockets, or None if it has none.

    `pip install uvicorn` does NOT pull in `websockets`; only
    `uvicorn[standard]` does. Without it uvicorn's `ws="auto"` resolves to
    "none" and a handshake is handled as an ordinary GET - which the auth
    middleware answers with a 303 to the login page. Nothing errors: the
    server is healthy, the page loads, and every camera panel just sits on
    "Mavjud emas" forever. That is what /faceid did in production, and it
    stayed hidden for weeks because a missing optional dependency is silent.
    """
    import importlib.util
    for name in ("websockets", "wsproto"):
        if importlib.util.find_spec(name) is not None:
            return name
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before anything is served: make sure the console is reachable. On a fresh
    # database this creates the default admin; once any account exists it does
    # nothing, so a changed password or a deleted default is never resurrected.
    try:
        if ensure_default_admin():
            log.warning("created the default admin account - change its password at /users")
    except Exception:
        log.exception("could not ensure a default admin account")
    if websocket_transport() is None:
        log.warning(
            "no WebSocket transport installed (neither `websockets` nor "
            "`wsproto`), so /ws/... cannot be served: uvicorn will answer the "
            "handshake as plain HTTP and the live camera panels will fall "
            "back to MJPEG at /video/<id>. Fix with: pip install --no-deps "
            "websockets")
    runtime.start()
    sweep = (asyncio.create_task(_eod_sweep_loop())
             if settings.eod_sweep_enabled else None)
    yield
    if sweep is not None:
        sweep.cancel()
    runtime.stop()


app = FastAPI(title="EmAtSy v3", version="3.0.0", lifespan=lifespan)

# Everything hangs off the configured prefix. The edge proxy forwards the path
# unchanged, so when this is served at /faceid the app must really answer on
# /faceid/... - mounts, routes and all.
PREFIX = settings.url_prefix.rstrip("/")

class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


settings.media_dir.mkdir(parents=True, exist_ok=True)
app.mount(f"{PREFIX}/media", StaticFiles(directory=str(settings.media_dir)), name="media")

# The original UI's stylesheet and scripts.
_static = settings.root / "static"
if _static.is_dir():
    app.mount(f"{PREFIX}/static", DevStaticFiles(directory=str(_static)), name="static")



@app.get(f"{PREFIX}/favicon.ico", include_in_schema=False)
def favicon():
    """The AIRI mark, served where a browser looks for it by itself.

    Both page templates declare it explicitly, which is what makes it work
    under a URL prefix - a bare /favicon.ico at the domain root belongs to
    the proxy, not to this app. This route answers the request a browser
    makes anyway when it has not parsed a page yet, and it is why
    `/favicon.ico` is in the public list in app/api/auth.py.
    """
    return FileResponse(_static / "favicon.ico", media_type="image/x-icon",
                        headers={"Cache-Control": "public, max-age=86400"})


# HTML pages (original Bootstrap templates) and the live-view WebSocket.
# Deny by default. This must be added BEFORE the routers so that every route,
# including any added later, is behind it unless explicitly listed public in
# app/api/auth.py. Without it the pages below answer anyone with the URL.
app.middleware("http")(auth_router.auth_middleware)

app.include_router(auth_router.router, prefix=PREFIX)
app.include_router(pages_router.router, prefix=PREFIX)
app.include_router(ws_router.router, prefix=PREFIX)


@app.get(f"{PREFIX}/health", include_in_schema=False)
def _health_probe():
    """Unauthenticated liveness probe for the edge proxy's monitoring.

    Deliberately says nothing about cameras, people or the gallery - it exists
    to answer "is the process up", and anything richer would leak operational
    detail to an endpoint that has to stay public.
    """
    return {"status": "ok"}


def _local(dt):
    return dt.astimezone(settings.tz) if dt else None


def _today() -> date:
    return business_date(datetime.now(settings.tz))


# --------------------------------------------------------------- stream ----
@app.get(f"{PREFIX}/video/{{camera_id}}")
async def video(camera_id: int):
    w = runtime.workers.get(camera_id)
    if w is None:
        raise HTTPException(404, "camera not running")

    async def gen():
        # `render()` resizes a 4K frame and JPEG-encodes it - tens of
        # milliseconds. Called directly in this generator it ran ON THE EVENT
        # LOOP, ten times a second per viewer, stalling every other request,
        # WebSocket send and page render in the process. The WebSocket path
        # (app/api/ws.py) always got this right; this one did not.
        while True:
            if w.source.is_stale:
                # Not repainted - see ws.frame_message. Multipart has no way
                # to say "still nothing", so the stream simply pauses and the
                # viewer keeps the last frame it received.
                await asyncio.sleep(1 / 10)
                continue
            # Shared with the WebSocket viewers: one encode per frame, not
            # one per viewer.
            jpg = await asyncio.to_thread(ws_router.shared_jpeg, w)
            if jpg:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(1 / 10)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


# ------------------------------------------------------------------ api ----
@app.get(f"{PREFIX}/api/health")
def health():
    ok = all(w.source.connected and not w.source.is_stale for w in runtime.workers.values())
    errs = sum(w.pipeline_errors for w in runtime.workers.values())
    # Replay has real gaps between clips, so `is_stale` goes true between
    # passes. Reporting that as "degraded" would be misleading, and reporting
    # it as "healthy" would be untrue - so it is named for what it is. A
    # pipeline error still degrades it, in replay as in production.
    if settings.replay_dir:
        status = "degraded" if errs else "replay"
    else:
        status = ("degraded" if errs else "healthy") if ok and runtime.workers else "degraded"
    return {
        "status": status,
        "replay": str(settings.replay_dir) if settings.replay_dir else None,
        "pipeline_errors": errs,
        "reid": runtime.reid.stats() if runtime.reid else None,
        "gallery": {"embeddings": len(runtime.gallery or []),
                    "people": runtime.gallery.n_people if runtime.gallery else 0},
        "cameras": [w.stats() for w in runtime.workers.values()],
    }


@app.get(f"{PREFIX}/api/events")
def events(limit: int = 40):
    # Snapshots leave as URLs carrying the deployment prefix, the same way
    # every rendered page hands them out.
    return [live_event(e) for e in runtime.events(limit)]


def _filtered_attendance_rows(day: str | None = None, query: str | None = None,
                              department: str | None = None, start_date: str | None = None,
                              end_date: str | None = None, status: str | None = None):
    """Fetch CSV/API attendance using the same predicates as the list page."""
    selected_day, start, end, _ = pages_router._attendance_date_range(day, start_date, end_date)
    with session_scope() as s:
        rows = s.execute(
            pages_router.attendance_newest_first(pages_router.attendance_records_query(
                start, end, query=query, department=department, status=status,
            ))
        ).all()
    return selected_day, start, end, [{
            "date": r.business_date.isoformat(),
            "employee_id": e.id, "name": e.full_name, "department": e.department,
            "check_in": _local(r.check_in_time).isoformat() if r.check_in_time else None,
            "check_out": _local(r.check_out_time).isoformat() if r.check_out_time else None,
            "worked_hours": round((r.worked_seconds or 0) / 3600.0, 2),
            "presence": r.presence.value, "status": r.status, "events": r.event_count,
        } for r, e in rows]


@app.get(f"{PREFIX}/api/attendance")
def attendance(day: str | None = None, query: str | None = None,
               department: str | None = None, start_date: str | None = None,
               end_date: str | None = None, status: str | None = None):
    _, _, _, rows = _filtered_attendance_rows(
        day, query, department, start_date, end_date, status,
    )
    return rows


@app.get(f"{PREFIX}/api/attendance/export")
def export(day: str | None = None, query: str | None = None,
           department: str | None = None, start_date: str | None = None,
           end_date: str | None = None, status: str | None = None):
    selected_day, start, end, rows = _filtered_attendance_rows(
        day, query, department, start_date, end_date, status,
    )
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["Date", "Employee", "Department", "Check in", "Check out", "Worked hours", "Status"])
    for r in rows:
        wr.writerow([
            r["date"], r["name"], r["department"],
            r["check_in"][11:16] if r["check_in"] else "",
            r["check_out"][11:16] if r["check_out"] else "",
            f'{r["worked_hours"]:.2f}', r["status"],
        ])
    buf.seek(0)
    suffix = selected_day.isoformat() if start == end else f"{start}_{end}"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=attendance_{suffix}.csv"})


@app.get(f"{PREFIX}/api/debug/captures")
def debug_captures():
    """What the debug folder holds, per person."""
    merged: dict[str, int] = {}
    for w in runtime.workers.values():
        for k, v in w.debug.summary().items():
            merged[k] = merged.get(k, 0) + v
    return {"dir": str(settings.debug_dir), "enabled": settings.debug_capture,
            "captured": dict(sorted(merged.items(), key=lambda kv: -kv[1]))}


@app.post(f"{PREFIX}/api/gallery/reload")
def reload_gallery():
    g = runtime.reload_gallery()
    return {"embeddings": len(g), "people": g.n_people}
