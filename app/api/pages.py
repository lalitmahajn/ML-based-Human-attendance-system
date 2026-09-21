"""HTML routes rendering the project's original Bootstrap templates.

These serve `templates/` — the UI the project shipped with — fed from the v3
recognition core.  `app/web/django_compat.py` supplies the Django constructs the
markup needs; `app/web/viewmodels.py` maps the v3 schema into the field names it
reads.
"""
from __future__ import annotations

from app.web.django_compat import media_path

import logging
from datetime import date, datetime, timedelta
from urllib.parse import quote as _quote, urlsplit, urlunsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from sqlalchemy import delete, func, or_, select
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import FormParser, MultiPartException, MultiPartParser

from app.config import settings
from app.db.models import (
    Camera, DailyAttendance, Employee, FaceEmbedding, PresenceStatus,
    PseudoPerson, RecognitionEvent, ReidPass, UnknownSighting,
)
from app.db.session import session_scope
from app.runtime import runtime
from app.services.attendance import business_date
from app.services.enrollment import (
    Enroller,
    EnrollmentImage,
    EnrollmentCaptureError,
    MAX_BROWSER_CAPTURES,
    MAX_CAPTURE_BYTES,
    MAX_ENROLLMENT_BODY_BYTES,
    MAX_ENROLLMENT_FIELD_BYTES,
    MAX_ENROLLMENT_FIELDS,
    decode_browser_captures,
    decode_uploaded_capture,
    enroll_employee_captures,
)
from app.web.django_compat import build_env
from app.web.viewmodels import DailyVM, EmployeeVM, EventVM, Page

log = logging.getLogger(__name__)


def _p(path: str) -> str:
    """A redirect target carrying the deployment prefix.

    Under /faceid a bare "/gallery/review" resolves against the
    DOMAIN root, which on the shared host belongs to another
    project - the same bug that has bitten the media and socket
    URLs here before.
    """
    return f"{settings.url_prefix.rstrip(chr(47))}{path}"
router = APIRouter(tags=["pages"])
env = build_env()


def render(name: str, *, request=None, current_view: str = "", **ctx) -> HTMLResponse:
    """Render a page with the common navigation, date, and signed-in user."""
    ctx.setdefault("request", request)
    ctx.setdefault("current_view", current_view)
    ctx.setdefault("today", today())
    # The nav needs to know who is signed in to show their name, the sign-out
    # control, and the admin-only Users link. Injected once here rather than
    # threaded through every page function.
    ctx.setdefault("current_user", getattr(getattr(request, "state", None), "user", None))
    # For the handful of links templates build by hand - static assets, form
    # actions, WebSocket URLs - which url() cannot reverse.
    ctx.setdefault("PREFIX", settings.url_prefix.rstrip("/"))
    return HTMLResponse(env.get_template(name).render(**ctx))


def today() -> date:
    return business_date(datetime.now(settings.tz))


def _recorded_presence_clause():
    """Attendance counts only after either an IN or OUT transition is recorded."""
    return or_(
        DailyAttendance.check_in_time.is_not(None),
        DailyAttendance.check_out_time.is_not(None),
    )


def _daily_rows(s, day: date, *, recorded_only: bool = False,
                limit: int | None = None, recent_first: bool = False):
    """Fetch daily rows; optional filtering/bounding is reserved for dashboard views."""
    query = (
        select(DailyAttendance, Employee)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(DailyAttendance.business_date == day)
    )
    if recorded_only:
        query = query.where(_recorded_presence_clause())
    if recent_first:
        query = query.order_by(
            func.coalesce(DailyAttendance.check_out_time, DailyAttendance.check_in_time).desc(),
            DailyAttendance.id.desc(),
        )
    else:
        query = query.order_by(DailyAttendance.check_in_time)
    if limit is not None:
        query = query.limit(limit)
    rows = s.execute(query).all()
    return [DailyVM.of(d, e) for d, e in rows]


# The dashboard is a recent operational summary, not the full attendance register.
DASHBOARD_DAILY_ROWS_LIMIT = 50


def _dashboard_daily_rows(s, day: date):
    """Return at most 50 newest recorded rows; `/attendance` intentionally remains full."""
    return _daily_rows(
        s, day, recorded_only=True, limit=DASHBOARD_DAILY_ROWS_LIMIT, recent_first=True,
    )


def _dashboard_attendance_metrics(s, day: date) -> tuple[int, int, int]:
    """Count all active recorded attendance independently from the bounded table slice."""
    filters = (
        DailyAttendance.business_date == day,
        Employee.is_active.is_(True),
        _recorded_presence_clause(),
    )
    present = s.execute(
        select(func.count(DailyAttendance.id))
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(*filters)
    ).scalar() or 0
    check_in_times = s.execute(
        select(DailyAttendance.check_in_time)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(*filters, DailyAttendance.check_in_time.is_not(None))
    ).scalars().all()
    late = sum(
        1 for check_in in check_in_times
        if (check_in.astimezone(settings.tz).hour, check_in.astimezone(settings.tz).minute) > (9, 0)
    )
    return present, len(check_in_times) - late, late


def _events(s, limit: int = 12):
    rows = s.execute(
        select(RecognitionEvent, Employee.full_name, Employee.department, Camera.name)
        .join(Employee, Employee.id == RecognitionEvent.employee_id)
        .outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
        .order_by(RecognitionEvent.ts.desc()).limit(limit)
    ).all()
    return [EventVM.of(e, n, d, c) for e, n, d, c in rows]


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _previous_months(value: date, count: int) -> list[date]:
    """Return calendar-month starts ending with the month containing ``value``."""
    cursor = _month_start(value)
    months = []
    for _ in range(count):
        months.append(cursor)
        cursor = (cursor - timedelta(days=1)).replace(day=1)
    return list(reversed(months))


def _dashboard_chart_data(s, day: date) -> tuple[list[dict], list[dict]]:
    """Build complete chart series with one grouped query per chart period."""
    week_start = day - timedelta(days=6)
    daily_counts = dict(s.execute(
        select(DailyAttendance.business_date, func.count(DailyAttendance.id))
        .where(
            DailyAttendance.business_date.between(week_start, day),
            _recorded_presence_clause(),
        )
        .group_by(DailyAttendance.business_date)
    ).all())
    weekly = [
        {"label": (week_start + timedelta(days=offset)).strftime("%d.%m"),
         "present": daily_counts.get(week_start + timedelta(days=offset), 0)}
        for offset in range(7)
    ]

    month_starts = _previous_months(day, 6)
    month_key = func.strftime("%Y-%m", DailyAttendance.business_date)
    monthly_counts = dict(s.execute(
        select(month_key, func.count(DailyAttendance.id))
        .where(
            DailyAttendance.business_date.between(month_starts[0], day),
            _recorded_presence_clause(),
        )
        .group_by(month_key)
    ).all())
    monthly = [
        {"label": month.strftime("%m.%Y"), "present": monthly_counts.get(month.strftime("%Y-%m"), 0)}
        for month in month_starts
    ]
    return weekly, monthly


def _camera_health(cameras: list[Camera]) -> list[dict]:
    """Describe enabled configured cameras without changing worker lifecycle."""
    worker_stats = {camera_id: worker.stats() for camera_id, worker in runtime.workers.items()}
    health = []
    for camera in cameras:
        stats = worker_stats.get(camera.id)
        if stats is None:
            health.append({
                "id": camera.id, "name": camera.name, "role": camera.role.value,
                "state": "offline", "state_label": "Offline", "available": False,
                "detail": "Worker not available", "fps": None, "pipeline_errors": None,
            })
            continue

        stream = stats.get("stream") or {}
        connected = bool(stream.get("connected")) and not bool(stream.get("stale"))
        health.append({
            "id": camera.id, "name": camera.name, "role": camera.role.value,
            "state": "online" if connected else "unavailable",
            "state_label": "Online" if connected else "Unavailable",
            "available": connected,
            "detail": "Stream active" if connected else "Stream not connected or stale",
            "fps": stream.get("fps") if connected else None,
            "pipeline_errors": stats.get("pipeline_errors", 0),
        })
    return health


# ---------------------------------------------------------------- dashboard --
@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, msg: str = "", error: str = ""):
    day = today()
    with session_scope() as s:
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        present, on_time, late = _dashboard_attendance_metrics(s, day)
        records = _dashboard_daily_rows(s, day)
        events = _events(s, 12)
        weekly_chart_data, monthly_chart_data = _dashboard_chart_data(s, day)
        cameras = s.execute(select(Camera).where(Camera.enabled.is_(True))
                            .order_by(Camera.name)).scalars().all()

    camera_health = _camera_health(cameras)

    return render(
        "dashboard/index.html", request=request, current_view="dashboard:home",
        total_employees=total, present_today=present, on_time_today=on_time,
        absent_today=max(0, total - present), late_arrivals=late,
        present_ratio=round(present / total * 100) if total else 0,
        today_attendance=records, recent_events=events,
        can_correct=bool(_can_correct(request)),
        can_admin=bool(_admin_only(request)), msg=msg, error=error,
        camera_health=camera_health, weekly_chart_data=weekly_chart_data,
        monthly_chart_data=monthly_chart_data,
        today=day,
    )


# ---------------------------------------------------------------- employees --
@router.get("/employees", response_class=HTMLResponse)
def employees_list(request: Request, query: str | None = None, department: str | None = None):
    with session_scope() as s:
        enrollment_counts = (
            select(
                FaceEmbedding.employee_id.label("employee_id"),
                func.count(FaceEmbedding.id).label("enrollment_count"),
            )
            .group_by(FaceEmbedding.employee_id)
            .subquery()
        )
        q = (
            select(
                Employee,
                func.coalesce(enrollment_counts.c.enrollment_count, 0).label("enrollment_count"),
            )
            .outerjoin(enrollment_counts, enrollment_counts.c.employee_id == Employee.id)
            .where(Employee.is_active.is_(True))
        )
        if query:
            q = q.where(or_(
                Employee.full_name.ilike(f"%{query}%"),
                Employee.external_id.ilike(f"%{query}%"),
            ))
        if department:
            q = q.where(Employee.department == department)
        emps = [EmployeeVM.of(employee, enrollment_count=count) for employee, count in
                s.execute(q.order_by(Employee.full_name)).all()]
        depts = [d for (d,) in s.execute(
            select(Employee.department)
            .where(Employee.is_active.is_(True))
            .distinct()
            .order_by(Employee.department)) if d]
    return render("employees/list.html", request=request, current_view="employees:list", employees=emps,
                  page_obj=Page(emps), is_paginated=False, query=query or "",
                  department=department or "", departments=depts)


# Declared before /employees/{employee_id}: a path param would otherwise
# swallow "add" and fail int conversion with a 422.
def _employee_registration_page(
    request: Request,
    *,
    employee=None,
    enrollment_error: str = "",
    enrollment_rejections=(),
    enrollment_success: str = "",
    enrollment_warning: str = "",
    enrollment_redirect_url: str = "",
    status_code: int = 200,
) -> HTMLResponse:
    with session_scope() as s:
        cams = [{"id": c.id, "label": c.name, "url": c.rtsp_url, "role": c.role.value}
                for c in s.execute(select(Camera)).scalars()]
    response = render(
        "employees/register.html",
        request=request,
        current_view="employees:register",
        employee=employee,
        action="add",
        cameras=cams,
        available_cameras=cams,
        # Always the browser's own camera. The IP-camera branch of the template
        # opens /ws/camera/registration/, a name app/api/ws.py has never
        # resolved (it knows ids, "primary", and camera names or roles), so on
        # every deployment with cameras configured - i.e. every real one - the
        # capture never produced a frame.
        use_ip_camera=False,
        default_camera_url=cams[0]["url"] if cams else "",
        enrollment_error=enrollment_error,
        enrollment_rejections=enrollment_rejections,
        enrollment_success=enrollment_success,
        enrollment_warning=enrollment_warning,
        enrollment_redirect_url=enrollment_redirect_url,
    )
    response.status_code = status_code
    return response


@router.get("/employees/add", response_class=HTMLResponse)
def employee_add(request: Request):
    return _employee_registration_page(request)


def _enrollment_wants_json(request: Request) -> bool:
    return "application/json" in request.headers.get("accept", "").lower()


ENROLLMENT_LIMIT_MESSAGE = (
    "Request limit exceeded. At most 32 images total, max 4 MB each, "
    "and total payload size must not exceed 10 MB."
)
ENROLLMENT_PERSISTENCE_MESSAGE = (
    "Could not save employee. Please try again; if problem persists "
    "contact administrator."
)


class EnrollmentRequestTooLarge(MultiPartException):
    """A bounded enrollment body exceeded a documented request limit."""


class EnrollmentRequestMalformed(ValueError):
    """The request body could not be interpreted as an enrollment form."""


class _EnrollmentMultiPartParser(MultiPartParser):
    """Starlette parser with a file-part cap enforced before spooled writes."""

    def __init__(self, *args, max_file_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_file_size = max_file_size
        self._current_file_size = 0

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._current_file_size = 0

    def on_headers_finished(self) -> None:
        try:
            super().on_headers_finished()
        except MultiPartException as exc:
            if str(exc).startswith("Too many"):
                raise EnrollmentRequestTooLarge(ENROLLMENT_LIMIT_MESSAGE) from exc
            raise

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        part_size = end - start
        if self._current_part.file is not None:
            self._current_file_size += part_size
            if self._current_file_size > self.max_file_size:
                raise EnrollmentRequestTooLarge(ENROLLMENT_LIMIT_MESSAGE)
        try:
            super().on_part_data(data, start, end)
        except MultiPartException as exc:
            if "maximum size" in str(exc):
                raise EnrollmentRequestTooLarge(ENROLLMENT_LIMIT_MESSAGE) from exc
            raise


async def _limited_enrollment_stream(request: Request):
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_ENROLLMENT_BODY_BYTES:
            raise EnrollmentRequestTooLarge(ENROLLMENT_LIMIT_MESSAGE)
        yield chunk


async def _parse_enrollment_form(request: Request) -> FormData:
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise EnrollmentRequestMalformed("Invalid Content-Length.") from exc
        if content_length < 0:
            raise EnrollmentRequestMalformed("Invalid Content-Length.")
        if content_length > MAX_ENROLLMENT_BODY_BYTES:
            raise EnrollmentRequestTooLarge(ENROLLMENT_LIMIT_MESSAGE)

    content_type = request.headers.get("content-type", "").lower()
    stream = _limited_enrollment_stream(request)
    if content_type.startswith("multipart/form-data"):
        parser = _EnrollmentMultiPartParser(
            request.headers,
            stream,
            max_files=MAX_BROWSER_CAPTURES,
            max_fields=MAX_ENROLLMENT_FIELDS,
            max_part_size=MAX_ENROLLMENT_FIELD_BYTES,
            max_file_size=MAX_CAPTURE_BYTES,
        )
        return await parser.parse()
    if content_type.startswith("application/x-www-form-urlencoded"):
        form = await FormParser(request.headers, stream).parse()
        if len(form.multi_items()) > MAX_ENROLLMENT_FIELDS:
            raise EnrollmentRequestTooLarge(ENROLLMENT_LIMIT_MESSAGE)
        return form
    raise EnrollmentRequestMalformed(
        "Invalid form format. Refresh the page and try submitting images again."
    )


def _enrollment_error_response(
    request: Request,
    message: str,
    *,
    status_code: int,
    employee=None,
    rejections=(),
):
    if _enrollment_wants_json(request):
        payload = {"ok": False, "error": message}
        if rejections:
            payload["rejected"] = list(rejections)
        return JSONResponse(payload, status_code=status_code)
    return _employee_registration_page(
        request,
        employee=employee,
        enrollment_error=message,
        enrollment_rejections=rejections,
        status_code=status_code,
    )


def _form_text(form: FormData, name: str) -> str:
    value = form.get(name, "")
    return value if isinstance(value, str) else ""


async def _decode_enrollment_form_images(form: FormData) -> list[EnrollmentImage]:
    all_files = [item for _, item in form.multi_items() if isinstance(item, UploadFile)]
    try:
        captures = decode_browser_captures(
            _form_text(form, "captured_images"),
            _form_text(form, "captured_image"),
        )
        camera_uploads = [
            item for item in form.getlist("camera_images") if isinstance(item, UploadFile)
        ]
        for index, uploaded in enumerate(camera_uploads, start=1):
            raw = await uploaded.read()
            captures.append(decode_uploaded_capture(
                raw,
                filename=uploaded.filename or f"camera-{index:03d}.jpg",
                content_type=uploaded.content_type or "",
                index=index,
                source_file=f"browser/{index:03d}.jpg",
            ))

        upload_files = [
            item for item in form.getlist("uploaded_images") if isinstance(item, UploadFile)
        ]
        for index, uploaded in enumerate(upload_files, start=1):
            raw = await uploaded.read()
            captures.append(decode_uploaded_capture(
                raw,
                filename=uploaded.filename or f"image-{index}",
                content_type=uploaded.content_type or "",
                index=index,
            ))
        return captures
    finally:
        for uploaded in all_files:
            await uploaded.close()


@router.post("/api/employees/")
async def employee_enroll(request: Request):
    """Create a v3 employee and gallery embeddings from browser captures."""
    try:
        form = await _parse_enrollment_form(request)
    except EnrollmentRequestTooLarge as exc:
        return _enrollment_error_response(request, str(exc), status_code=413)
    except (EnrollmentRequestMalformed, MultiPartException) as exc:
        log.warning("invalid enrollment request: %s", exc)
        return _enrollment_error_response(
            request,
            "Could not read form data. Refresh the page and try again.",
            status_code=400,
        )
    except Exception:
        log.exception("unexpected enrollment multipart parsing failure")
        return _enrollment_error_response(
            request,
            "Could not read form data. Refresh the page and try again.",
            status_code=400,
        )

    form_values = {
        name: _form_text(form, name)
        for name in ("full_name", "position", "department", "phone_number", "email", "notes")
    }
    try:
        captures = await _decode_enrollment_form_images(form)
        enroller = await run_in_threadpool(Enroller)
        result = await run_in_threadpool(
            enroll_employee_captures,
            full_name=form_values["full_name"],
            position=form_values["position"],
            department=form_values["department"],
            phone_number=form_values["phone_number"],
            captures=captures,
            enroller=enroller,
        )
    except EnrollmentCaptureError as exc:
        message = str(exc)
        rejections = [
            {"source_file": item.source_file, "error": item.error}
            for item in exc.rejections
        ]
        return _enrollment_error_response(
            request,
            employee=form_values,
            message=message,
            rejections=rejections,
            status_code=422,
        )
    except Exception:
        log.exception("native v3 employee enrollment persistence failed")
        return _enrollment_error_response(
            request,
            ENROLLMENT_PERSISTENCE_MESSAGE,
            employee=form_values,
            status_code=500,
        )

    warning = ""
    try:
        gallery = await run_in_threadpool(runtime.reload_gallery)
        gallery_summary = {"embeddings": len(gallery), "people": gallery.n_people}
    except Exception:
        log.exception("employee %s saved but runtime gallery reload failed", result.employee_id)
        gallery_summary = None
        warning = (
            "Employee saved, but runtime gallery was not reloaded. "
            "Please restart the recognition service."
        )

    redirect_url = _p(f"/employees/{result.employee_id}")
    if _enrollment_wants_json(request):
        return JSONResponse(
            {
                "ok": True,
                "employee_id": result.employee_id,
                "external_id": result.external_id,
                "embeddings": result.embeddings,
                "redirect_url": redirect_url,
                "gallery": gallery_summary,
                "warning": warning,
                "rejected": [
                    {"source_file": item.source_file, "error": item.error}
                    for item in result.rejected
                ],
            },
            status_code=201,
        )
    if result.rejected or warning:
        return _employee_registration_page(
            request,
            employee=form_values,
            enrollment_rejections=[
                {"source_file": item.source_file, "error": item.error}
                for item in result.rejected
            ],
            enrollment_success=(
                f"Employee saved and {result.embeddings} embeddings added to gallery."
            ),
            enrollment_warning=warning,
            enrollment_redirect_url=redirect_url,
            status_code=201,
        )
    return HTMLResponse(
        f'<!doctype html><title>Employee Saved</title><a href="{redirect_url}">Employee Page</a>',
        status_code=303,
        headers={"Location": redirect_url},
    )


@router.get("/employees/{employee_id}", response_class=HTMLResponse)
def employee_detail(request: Request, employee_id: int):
    from app.services import augment as augment_svc
    with session_scope() as s:
        e = s.get(Employee, employee_id)
        if not e:
            raise HTTPException(404, "Employee not found")
        rows = s.execute(
            select(DailyAttendance).where(DailyAttendance.employee_id == employee_id)
            .order_by(DailyAttendance.business_date.desc()).limit(30)).scalars().all()
        hist = [DailyVM.of(d, e) for d in rows]
        n_emb = s.execute(
            select(func.count(FaceEmbedding.id)).where(FaceEmbedding.employee_id == employee_id)
        ).scalar() or 0
        has_photo = n_emb > 0 and augment_svc.profile_image(employee_id) is not None
        vm = EmployeeVM.of(
            e, enrollment_count=n_emb,
            image=_p(f"/employees/{employee_id}/photo") if has_photo else None)
    return render("employees/detail.html", request=request, current_view="employees:list", employee=vm,
                  attendance_history=hist, records=hist, embedding_count=n_emb)


@router.get("/employees/{employee_id}/delete", response_class=HTMLResponse)
def employee_delete_confirm(request: Request, employee_id: int):
    with session_scope() as s:
        emp = s.get(Employee, employee_id)
        if not emp:
            raise HTTPException(404, "Employee not found")
        return render("employees/delete.html", request=request, current_view="employees:list", object=emp)


@router.post("/employees/{employee_id}/delete")
async def employee_delete(request: Request, employee_id: int):
    with session_scope() as s:
        emp = s.get(Employee, employee_id)
        if not emp:
            raise HTTPException(404, "Employee not found")
        name = emp.full_name
        s.execute(delete(FaceEmbedding).where(FaceEmbedding.employee_id == employee_id))
        s.execute(delete(DailyAttendance).where(DailyAttendance.employee_id == employee_id))
        s.execute(delete(Employee).where(Employee.id == employee_id))

    try:
        await run_in_threadpool(runtime.reload_gallery)
    except Exception:
        log.exception("gallery reload after employee deletion failed")

    return RedirectResponse(_p(f"/employees?msg=" + _quote(f"Employee '{name}' deleted successfully.")), status_code=303)



@router.get("/employees/{employee_id}/photo")
def employee_photo(request: Request, employee_id: int):
    """The registered photograph shown on one person's profile.

    `face_id_users` is deliberately NOT under the public media mount, so this
    cannot be a static URL: it is served here, behind the session the rest of
    the page already needs, and containment-checked in `profile_image` because
    the id is caller input.

    Gated at "signed in", not at admin, unlike /gallery/enrolment. That route
    exists to browse the biometric gallery by embedding id; this one answers
    "what does the person on this profile look like" for a page every signed-in
    role can already open, and it can only ever return the one photograph
    belonging to the employee named in the path.
    """
    from app.services import augment
    p = augment.profile_image(employee_id)
    if p is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    # Private, not public: it is one person's face behind a session cookie, and
    # a shared cache would serve it to whoever asked next.
    return FileResponse(str(p), headers={"Cache-Control": "private, max-age=300"})


# --------------------------------------------------------------- attendance --
MAX_ATTENDANCE_RANGE_DAYS = 366
# The HTML table stops at a calendar month; the CSV export keeps the year.
MAX_ATTENDANCE_HTML_DAYS = 31


def _parse_attendance_date(value: str | None, field: str) -> date | None:
    """Parse a query date without allowing malformed values to escape as 500s."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"{field} must be an ISO date") from exc


def _attendance_date_range(target_date: str | None, start_date: str | None,
                           end_date: str | None) -> tuple[date, date, date, str | None]:
    """Return a bounded inclusive filter range and an optional normalization notice."""
    selected_day = _parse_attendance_date(target_date, "target_date") or today()
    start = _parse_attendance_date(start_date, "start_date")
    end = _parse_attendance_date(end_date, "end_date")
    if start is None and end is None:
        start = end = selected_day
    elif start is None:
        start = end
    elif end is None:
        end = start

    notice = None
    if start > end:
        start, end = end, start
        notice = "Date range was reordered."
    if (end - start).days >= MAX_ATTENDANCE_RANGE_DAYS:
        raise HTTPException(422, "Date range cannot exceed 366 days")
    return selected_day, start, end, notice


def attendance_records_query(start: date, end: date, *, query: str | None = None,
                             department: str | None = None, status: str | None = None):
    """Build the shared employee-attendance filter query used by HTML and CSV views."""
    statement = (
        select(DailyAttendance, Employee)
        .join(Employee, Employee.id == DailyAttendance.employee_id)
        .where(DailyAttendance.business_date.between(start, end))
    )
    if query:
        statement = statement.where(or_(
            Employee.full_name.ilike(f"%{query}%"),
            Employee.external_id.ilike(f"%{query}%"),
        ))
    if department:
        statement = statement.where(Employee.department == department)
    if status:
        statement = statement.where(DailyAttendance.status == status)
    return statement


def attendance_newest_first(statement):
    """Keep attendance reporting order deterministic across consumers."""
    return statement.order_by(
        DailyAttendance.business_date.desc(),
        func.coalesce(DailyAttendance.check_out_time, DailyAttendance.check_in_time).desc(),
        DailyAttendance.id.desc(),
    )


@router.get("/attendance", response_class=HTMLResponse)
def attendance_list(request: Request, target_date: str | None = None,
                    query: str | None = None, department: str | None = None,
                    start_date: str | None = None, end_date: str | None = None,
                    status: str | None = None):
    selected_day, start, end, range_notice = _attendance_date_range(
        target_date, start_date, end_date,
    )
    # The page materialises every row in the range - up to a year of them,
    # thousands of rows nobody scrolls. The table shows the most recent
    # month, says so, and points at the CSV export, which keeps the full
    # range (it reads the ORIGINAL query string). A difference of 31 days is
    # allowed so the page's own "Bir oy" shortcut never trips the note.
    cap_notice = ""
    if (end - start).days > MAX_ATTENDANCE_HTML_DAYS:
        start = end - timedelta(days=MAX_ATTENDANCE_HTML_DAYS)
        cap_notice = (f"The table displays at most {MAX_ATTENDANCE_HTML_DAYS} days: "
                      f"{start.isoformat()} — {end.isoformat()}. For the full range, "
                      f"use CSV export.")
    with session_scope() as s:
        records = [DailyVM.of(record, employee) for record, employee in s.execute(
            attendance_newest_first(attendance_records_query(
                start, end, query=query, department=department, status=status,
            ))
        ).all()]
        departments = [value for (value,) in s.execute(
            select(Employee.department)
            .where(Employee.is_active.is_(True))
            .distinct()
            .order_by(Employee.department)
        ) if value]

    return render(
        "attendance/list.html", request=request, current_view="attendance:list", records=records,
        daily_records=records, page_obj=Page(records), is_paginated=False,
        selected_date=selected_day, start_date=start.isoformat(), end_date=end.isoformat(),
        query=query or "", employee=query or "", department=department or "", status=status or "",
        departments=departments, range_notice=range_notice, cap_notice=cap_notice,
    )


@router.get("/attendance/history", response_class=HTMLResponse)
def attendance_history(request: Request):
    return attendance_list(request)


def _pseudo_for(s, sightings) -> dict[int, dict]:
    """Which pseudo-person each unknown sighting belongs to, if any.

    THE JOIN IS BY (camera, track, day) AND NOT BY ANYTHING BETTER, because
    there is nothing better: `unknown_sighting` and `reid_pass` are written by
    two different threads from the same completed track and share no key. That
    pair is unique within a business date except across a `tracker.reset()`,
    which restarts the counter - so the candidate whose `last_seen` is nearest
    wins, and a candidate more than a few minutes away is not used at all. A
    wrong group shown next to a face is worse than no group.

    Returns {sighting_id: {code, n_passes, employee_id, name, by}}.
    """
    if not sightings:
        return {}
    keys = {(u.camera_id, u.track_id) for u in sightings}
    days = {u.business_date for u in sightings}
    rows = s.execute(
        select(ReidPass, PseudoPerson)
        .join(PseudoPerson, PseudoPerson.id == ReidPass.pseudo_person_id)
        .where(ReidPass.business_date.in_(days),
               ReidPass.camera_id.in_({c for c, _t in keys}),
               ReidPass.track_id.in_({t for _c, t in keys}))
    ).all()
    if not rows:
        return {}
    by_key: dict[tuple, list] = {}
    for r, pp in rows:
        by_key.setdefault((r.camera_id, r.track_id), []).append((r, pp))

    names = dict(s.execute(
        select(Employee.id, Employee.full_name).where(
            Employee.id.in_({pp.employee_id for _r, pp in rows
                             if pp.employee_id is not None} or {-1}))).all())
    out: dict[int, dict] = {}
    for u in sightings:
        cand = by_key.get((u.camera_id, u.track_id))
        if not cand:
            continue
        r, pp = min(cand, key=lambda x: abs(
            (x[0].last_seen - u.last_seen).total_seconds()))
        if abs((r.last_seen - u.last_seen).total_seconds()) > 180:
            continue
        out[u.id] = {"code": pp.code, "n_passes": pp.n_passes or 0,
                     "employee_id": pp.employee_id,
                     "name": names.get(pp.employee_id or -1, ""),
                     "by": r.pseudo_by or ""}
    return out


@router.get("/attendance/unknown", response_class=HTMLResponse)
def attendance_unknown(request: Request, show: str = "open", msg: str = "",
                       error: str = ""):
    """The review queue, and - for an admin - the place labels come from.

    `show=open` hides what has already been decided, because a queue that never
    shrinks stops being reviewed. `show=all` brings the decisions back so they
    can be corrected.
    """
    me = _can_correct(request)
    with session_scope() as s:
        q = select(UnknownSighting).order_by(UnknownSighting.last_seen.desc())
        if show != "all":
            q = q.where(UnknownSighting.resolved_kind.is_(None))
        rows = s.execute(q.limit(100)).scalars().all()
        names = dict(s.execute(
            select(Employee.id, Employee.full_name)
            .where(Employee.is_active.is_(True))
            .order_by(Employee.full_name)).all())
        groups = _pseudo_for(s, rows)
        attempts = [{
            "id": u.id, "camera_id": u.camera_id, "attempt_count": u.frames,
            "first_seen": u.first_seen.astimezone(settings.tz),
            "last_seen": u.last_seen.astimezone(settings.tz),
            "best_score": u.best_score or 0.0,
            "resolved_kind": u.resolved_kind,
            "resolved_employee_id": u.resolved_employee_id,
            "resolved_name": names.get(u.resolved_employee_id or -1, ""),
            "resolved_by": u.resolved_by or "",
            "has_vector": u.vector is not None,
            # Which pseudo-person this face was grouped into, so a reviewer
            # sees "this is the fourth time we have seen this person" instead
            # of one anonymous card among a hundred. It is a HINT and is
            # labelled as one: the grouping recalls about 30% of a person's
            # passes, so a card with no group is not evidence of a first visit.
            "pseudo": groups.get(u.id),
            "latest_record": {"snapshot": {"url": media_path(u.snapshot) if u.snapshot else None}},
        } for u in rows]
    return render("attendance/unknown.html", request=request, current_view="attendance:unknown",
                  unknown_attempts=attempts, page_obj=Page(attempts), is_paginated=False,
                  employees=sorted(names.items(), key=lambda kv: kv[1]),
                  can_correct=bool(me), can_admin=bool(_admin_only(request)),
                  show=show, msg=msg, error=error)


def _capture_index() -> dict[str, str]:
    """Debug-capture stems on disk, keyed by everything but their sequence.

    `corrections.capture_stem` finds one event's capture with a glob over
    data/debug. Called per event, a day view is a directory walk per row, so
    the day view lists the directory ONCE and looks its events up here. The
    key is the stem minus its trailing `_NNN` sequence - exactly the prefix
    `capture_stem` globs for - and the first hit in path order wins, as it
    does there.
    """
    import glob
    from pathlib import Path
    index: dict[str, str] = {}
    for hit in sorted(glob.glob(str(settings.debug_dir / "*" / "*.json"))):
        stem = Path(hit).stem
        index.setdefault(stem[:stem.rfind("_") + 1], stem)
    return index


def _capture_stem_from(index: dict[str, str], ts, score, camera: str) -> str:
    """`corrections.capture_stem`, answered from a prebuilt index."""
    if not ts:
        return ""
    local = ts.astimezone(settings.tz)
    return index.get(f"{local:%Y%m%d_%H%M%S}_{float(score or 0):.3f}_{camera or ''}_", "")


@router.get("/attendance/day/{employee_id}/{day}", response_class=HTMLResponse)
def attendance_day(request: Request, employee_id: int, day: str, msg: str = "",
                   error: str = ""):
    """Every event behind one person's day, with the face that decided each.

    The dashboard feed shows the last twelve recognitions, which is no use for
    a wrong check-out found the next morning. This is the view that makes a
    correction possible: the whole day, in order, each event beside the crop
    the recognizer actually saw, and - for an admin - a button to void it.

    The BODY crop stored on the event is not enough to judge by. It shows who
    walked past; it does not show what was matched. A false accept is only
    visible when the face is on screen next to the name it was given.
    """
    bdate = _parse_attendance_date(day, "day")
    if bdate is None:
        raise HTTPException(400, "bad date")
    me = _can_correct(request)
    with session_scope() as s:
        emp = s.get(Employee, employee_id)
        if emp is None:
            raise HTTPException(404, "Employee not found")
        daily = s.execute(select(DailyAttendance).where(
            DailyAttendance.employee_id == employee_id,
            DailyAttendance.business_date == bdate)).scalar_one_or_none()
        rows = s.execute(
            select(RecognitionEvent, Camera.name)
            .outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
            .where(RecognitionEvent.employee_id == employee_id,
                   RecognitionEvent.business_date == bdate)
            .order_by(RecognitionEvent.ts)).all()
        events = []
        captures = _capture_index()
        for e, cam in rows:
            stem = _capture_stem_from(captures, e.ts, e.score, cam or "")
            events.append({
                "id": e.id, "time": e.ts.astimezone(settings.tz).strftime("%H:%M:%S"),
                "camera": cam or "—", "role": e.role.value if hasattr(e.role, "value") else str(e.role),
                "transition": e.transition or "", "direction": e.direction or "",
                "score": e.score or 0.0, "margin": e.margin or 0.0,
                "votes": e.votes or "", "voided": e.voided_at is not None,
                "manual": (e.source or "live") == "manual",
                "void_reason": e.void_reason or "", "voided_by": e.voided_by or "",
                "snapshot": media_path(e.snapshot) if e.snapshot else None,
                "stem": stem,
            })
        vm = {"id": emp.id, "name": emp.full_name, "department": emp.department or ""}
        summary = None
        if daily is not None:
            summary = {
                "check_in": daily.check_in_time.astimezone(settings.tz).strftime("%H:%M:%S")
                if daily.check_in_time else None,
                "check_out": daily.check_out_time.astimezone(settings.tz).strftime("%H:%M:%S")
                if daily.check_out_time else None,
                "worked": round((daily.worked_seconds or 0) / 3600.0, 2),
                "status": daily.status, "presence": daily.presence.value
                if hasattr(daily.presence, "value") else str(daily.presence),
            }
    return render("attendance/day.html", request=request, current_view="attendance:list",
                  employee=vm, day=bdate, events=events, summary=summary,
                  can_correct=bool(me), msg=msg, error=error)


@router.get("/attendance/event/{event_id}/evidence/{kind}")
def event_evidence(request: Request, event_id: int, kind: str):
    """The face or the full frame behind one recognition. Admin-gated.

    `data/debug` holds face images of identified people and is deliberately
    outside the public media mount, so this is gated and the resolved path is
    containment-checked - the same treatment /gallery/crop gets.
    """
    from app.services import corrections
    # Whoever may void the event must be able to look at what it captured -
    # a correction made without seeing the face is a guess.
    if not _can_correct(request):
        return JSONResponse({"detail": "Not permitted"}, status_code=403)
    if kind not in ("face", "aligned", "frame"):
        return JSONResponse({"detail": "bad kind"}, status_code=400)
    with session_scope() as s:
        row = s.execute(
            select(RecognitionEvent.ts, RecognitionEvent.score, Camera.name)
            .outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
            .where(RecognitionEvent.id == event_id)).first()
    if not row:
        return JSONResponse({"detail": "not found"}, status_code=404)
    got = corrections.evidence(corrections.capture_stem(row[0], row[1], row[2] or ""))
    if kind not in got:
        return JSONResponse({"detail": "no capture kept for this event"},
                            status_code=404)
    return FileResponse(str(got[kind]), media_type="image/jpeg")


# ----------------------------------------------------------- corrections ---
# An admin saying "that is not that person", and an admin saying who an unknown
# face actually was. Both fix the record AND leave a label behind - see
# app/services/corrections.py for why the labels are the more valuable half.

def _back(request: Request, form, default: str) -> str:
    """Where to return to after a correction.

    The target is caller input, so only a same-site absolute path is accepted.
    Anything else - a scheme, a protocol-relative "//host" - falls back to the
    default rather than becoming an open redirect out of an authenticated page.
    """
    want = str(form.get("next") or "").strip()
    if want.startswith("/") and not want.startswith("//") and "\\" not in want:
        return want
    return _p(default)


@router.post("/attendance/event/{event_id}/void")
async def event_void(request: Request, event_id: int):
    from app.services import corrections
    me = _can_correct(request)
    if not me:
        return JSONResponse({"detail": "Not permitted"}, status_code=403)
    form = await request.form()
    # Rebuilds the day and may recalibrate the gallery: database and numpy
    # work that must not run on the event loop, where it stalls every frame
    # on its way to every viewer. Same treatment as the enrol handler.
    out = await run_in_threadpool(
        corrections.void_event, event_id, by=str(me.get("u") or ""),
        reason=str(form.get("reason") or ""))
    if out.get("ok"):
        extra = (f" {out['gallery_rows_removed']} gallery frame(s) removed."
                 if out.get("gallery_rows_removed") else "")
        note = (f"Recalculated {out['business_date']} for {out['name']} "
                f"({out['replayed']} events).{extra}")
        target = f"{_back(request, form, '/')}?msg={_quote(note)}"
    else:
        target = f"{_back(request, form, '/')}?error={_quote(out.get('error', ''))}"
    return RedirectResponse(target, status_code=303)


@router.post("/attendance/event/{event_id}/unvoid")
async def event_unvoid(request: Request, event_id: int):
    from app.services import corrections
    me = _can_correct(request)
    if not me:
        return JSONResponse({"detail": "Not permitted"}, status_code=403)
    form = await request.form()
    out = await run_in_threadpool(corrections.unvoid_event, event_id,
                                  by=str(me.get("u") or ""))
    key = "msg" if out.get("ok") else "error"
    val = "Restored." if out.get("ok") else out.get("error", "")
    return RedirectResponse(f"{_back(request, form, '/')}?{key}={_quote(val)}",
                            status_code=303)


@router.post("/attendance/unknown/{sighting_id}/resolve")
async def unknown_resolve(request: Request, sighting_id: int):
    from app.services import corrections
    me = _can_correct(request)
    if not me:
        return JSONResponse({"detail": "Not permitted"}, status_code=403)
    form = await request.form()
    emp = str(form.get("employee_id") or "").strip()
    back = _back(request, form, "/attendance/unknown")
    direction = str(form.get("direction") or "").strip().upper()

    # Stating the direction is what turns a label into attendance, so it is
    # gated harder than labelling: authoring a pass is a strictly larger power
    # than naming a face, and an operator naming a face still cannot invent one.
    if direction:
        from app.services import auth as auth_svc
        if not auth_svc.can_admin(me):
            return RedirectResponse(
                f"{back}?error={_quote('Administrator permissions required to record attendance')}",
                status_code=303)
        out = await run_in_threadpool(
            corrections.promote_sighting, sighting_id,
            employee_id=int(emp) if emp.isdigit() else 0,
            direction=direction, by=str(me.get("u") or ""))
        if not out.get("ok"):
            return RedirectResponse(f"{back}?error={_quote(out.get('error', ''))}",
                                    status_code=303)
        moved = {"CHECK_IN": "IN", "CHECK_OUT": "OUT"}.get(
            out["transition"], out["transition"] or "record")
        note = (f"{out['name']} recorded to attendance as {out['direction']} ({moved}). "
                f"Marked as manually entered; if incorrect, void it from the day view.")
        return RedirectResponse(f"{back}?msg={_quote(note)}", status_code=303)

    out = await run_in_threadpool(
        corrections.resolve_sighting,
        sighting_id, kind=str(form.get("kind") or ""),
        employee_id=int(emp) if emp.isdigit() else None,
        by=str(me.get("u") or ""))
    if not out.get("ok"):
        return RedirectResponse(f"{back}?error={_quote(out.get('error', ''))}",
                                status_code=303)
    if out["kind"] == "employee":
        # Say the part the operator cannot see. Naming the face looks like it
        # finished the job - the row turns green with the person's name on it -
        # and the one thing it does NOT do is the thing they came here for.
        note = (f"Labeled as {out['name']}. Note: attendance was not modified &mdash; "
                f"labeling does not create an IN/OUT record.")
        if out.get("offerable"):
            note += (" To add this face to the gallery, visit the Gallery page.")
    elif out["kind"] == "visitor":
        note = ("Labeled as Not Employee. This is valuable data for threshold calibration.")
    else:
        note = "Labeled."
    return RedirectResponse(f"{back}?msg={_quote(note)}", status_code=303)


# ------------------------------------------------------------------ cameras --
def _safe_stream_url(value: str | None) -> str | None:
    """Keep a camera stream's host/path while never rendering its credentials."""
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return None
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _camera_diagnostics_context(cameras: list[Camera]) -> list[dict]:
    """Present configuration and worker observations without changing either system."""
    worker_stats = {}
    worker_failures = {}
    for camera_id, worker in runtime.workers.items():
        try:
            worker_stats[camera_id] = worker.stats()
        except Exception as exc:  # Diagnostics must not make camera failures fatal to the page.
            log.warning("camera %s stats unavailable: %s", camera_id, exc)
            worker_failures[camera_id] = "Worker statistics unavailable"
    diagnostics = []
    for camera in cameras:
        stats = worker_stats.get(camera.id)
        stream = (stats or {}).get("stream") or {}
        online = bool(stream.get("connected")) and not bool(stream.get("stale"))
        resolution = stream.get("resolution") if online else None
        if isinstance(resolution, str) and resolution.lower().replace("×", "x") == "0x0":
            resolution = None
        observed_fps = stream.get("fps") if online else None
        if isinstance(observed_fps, bool) or not isinstance(observed_fps, (int, float)) or observed_fps <= 0:
            observed_fps = None

        pipeline = stats or {}
        diagnostics.append({
            "configured": {
                "id": camera.id, "name": camera.name, "role": camera.role.value,
                "ip": camera.ip or None, "rtsp_url": _safe_stream_url(camera.rtsp_url),
                "enabled": camera.enabled,
            },
            "observed": {
                "state": "online" if online else "offline",
                "state_label": "Online" if online else "Unavailable",
                "resolution": resolution,
                "fps": observed_fps,
                "detail": (
                    "RTSP stream active" if online
                    else worker_failures.get(camera.id, "Worker or updated RTSP stream not available")
                ),
            },
            "pipeline": {
                "fps": _algorithm_fps(observed_fps, pipeline.get("algorithm_fps"), (pipeline.get("timings") or {}).get("total")),
                "latency_ms": _positive_number((pipeline.get("timings") or {}).get("total")),
                "errors": pipeline.get("pipeline_errors"),
                "last_error": pipeline.get("last_error"),
            },
            "drift": {
                "state": "nvr-owned", "comparison": "unavailable",
                "label": "Data comparison not available",
                "detail": "Encoder settings managed by NVR",
            },
        })
    return diagnostics


@router.get("/cameras", response_class=HTMLResponse)
def cameras(request: Request):
    with session_scope() as s:
        cams = s.execute(select(Camera)).scalars().all()
    diagnostics = _camera_diagnostics_context(cams)
    return render("camera/settings.html", request=request, current_view="camera:settings",
                  camera_diagnostics=diagnostics)


# Cameras are configured with scripts/ (see docs/OPERATIONS.md). The page that
# used to live at /cameras/rtsp posted to /api/cameras/add, which nothing has
# ever served, so it was a form that could only fail.


# -------------------------------------------------------------- recognition --
def _positive_number(value):
    """Return a positive numeric metric, excluding booleans and invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _algorithm_fps(stream_fps, actual_fps=None, latency_ms=None) -> float | None:
    """Return true observed algorithm processing FPS.

    Uses the worker's measured processing rate when available, or calculates
    exact throughput from frame latency (1000 / latency_ms). Falls back to
    paced stream rate if latency is uninitialized.
    """
    if actual_fps is not None and isinstance(actual_fps, (int, float)) and actual_fps > 0:
        return round(float(actual_fps), 1)
    lat = _positive_number(latency_ms)
    if lat is not None and lat > 0:
        return round(min(60.0, 1000.0 / lat), 1)
    fps = _positive_number(stream_fps)
    if fps is None:
        return None
    return round(fps / max(1, int(settings.process_every_nth or 1)), 1)


def _available_resolution(value) -> str | None:
    """Reject the uninitialized resolution emitted by a fresh RtspSource."""
    if not isinstance(value, str) or not value.strip():
        return None
    resolution = value.strip()
    if resolution.lower().replace("×", "x") == "0x0":
        return None
    return resolution


def _last_frame_label(value) -> str | None:
    """Format an observed worker timestamp without inventing one when it is absent."""
    if value is None:
        return None
    parsed = value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = value / 1000 if value > 10_000_000_000 else value
        parsed = datetime.fromtimestamp(timestamp, settings.tz)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    if isinstance(parsed, datetime):
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=settings.tz)
        return parsed.astimezone(settings.tz).strftime("%H:%M:%S")
    return str(parsed)


def _live_camera_context(cameras: list[Camera]) -> list[dict]:
    """Map configured cameras and existing worker stats into presentation-only health data."""
    role_order = {"IN": 0, "OUT": 1, "BOTH": 2}
    role_labels = {
        "IN": "Entrance Camera", "OUT": "Exit Camera", "BOTH": "General Camera",
    }
    contexts = []
    for camera in sorted(cameras, key=lambda item: (role_order.get(item.role.value, 3), item.name)):
        worker = runtime.workers.get(camera.id)
        stats = None
        if worker is not None:
            try:
                stats = worker.stats() or {}
            except Exception as exc:  # a broken diagnostics call must not hide the live page
                log.warning("camera %s stats unavailable: %s", camera.id, exc)

        role = camera.role.value
        if stats is None:
            contexts.append({
                "id": camera.id, "name": camera.name, "role": role,
                "role_label": role_labels.get(role, "Camera"),
                "state": "offline", "state_label": "Offline",
                "state_detail": "Worker not available",
                "resolution": None, "camera_fps": None, "algorithm_fps": None,
                "latency_ms": None, "last_frame": None,
                "pipeline_errors": None, "pipeline_status": "unavailable",
                "pipeline_status_label": "Pipeline status unavailable",
                "last_error": None,
            })
            continue

        stream = stats.get("stream") or {}
        timings = stats.get("timings") or {}
        connected = bool(stream.get("connected")) and not bool(stream.get("stale"))
        state = "online" if connected else "unavailable"
        state_label = "Online" if connected else "Unavailable"
        state_detail = "Stream active" if connected else "Stream not connected or stale"

        # Every figure here comes from what CameraWorker.stats() and the source
        # really carry: stream fps, the last frame's pipeline timings, and the
        # source's own last-frame clock. The keys this used to read
        # (processed_fps, latency_ms, dropped_frames, last_frame_time) did not
        # exist, so the live page showed "Unavailable" for all of them.
        last_frame_ts = _positive_number(
            getattr(getattr(worker, "source", None), "last_frame_ts", None))

        pipeline_errors = stats.get("pipeline_errors") if "pipeline_errors" in stats else None
        last_error = stats.get("last_error") or None
        if last_error or (isinstance(pipeline_errors, (int, float)) and pipeline_errors > 0):
            pipeline_status, pipeline_status_label = "error", "Pipeline Error"
        elif pipeline_errors == 0 and connected:
            pipeline_status, pipeline_status_label = "healthy", "Pipeline Healthy"
        else:
            pipeline_status, pipeline_status_label = "unavailable", "Pipeline status unavailable"

        contexts.append({
            "id": camera.id, "name": camera.name, "role": role,
            "role_label": role_labels.get(role, "Camera"),
            "state": state, "state_label": state_label, "state_detail": state_detail,
            "resolution": _available_resolution(stream.get("resolution")),
            "camera_fps": _positive_number(stream.get("fps")),
            "algorithm_fps": _algorithm_fps(stream.get("fps"), stats.get("algorithm_fps"), timings.get("total")),
            "latency_ms": _positive_number(timings.get("total")),
            "last_frame": _last_frame_label(last_frame_ts),
            "pipeline_errors": pipeline_errors, "pipeline_status": pipeline_status,
            "pipeline_status_label": pipeline_status_label, "last_error": last_error,
        })
    return contexts


def _event_payload(events: list[EventVM]) -> list[dict]:
    """Serialize quality-best recognition evidence for initial render and polling."""
    return [{
        "name": event.employee_name, "department": event.employee_department,
        "camera": event.camera, "action": event.action_type,
        "transition": event.transition, "score": round(event.confidence, 3),
        "time": event.timestamp.strftime("%H:%M:%S"), "snapshot": event.snapshot,
    } for event in events]


def _unknown_activity(s, day: date, limit: int = 30) -> list[dict]:
    """Return newest quality-best unknown sightings with camera presentation data."""
    rows = s.execute(
        select(UnknownSighting, Camera.name)
        .outerjoin(Camera, Camera.id == UnknownSighting.camera_id)
        .where(UnknownSighting.business_date == day)
        .order_by(UnknownSighting.last_seen.desc(), UnknownSighting.id.desc())
        .limit(limit)
    ).all()
    return [{
        "id": sighting.id, "camera_id": sighting.camera_id,
        "camera": camera_name or (f"Camera #{sighting.camera_id}" if sighting.camera_id else "—"),
        "attempt_count": sighting.frames or 0,
        "first_seen": sighting.first_seen.astimezone(settings.tz).isoformat(),
        "last_seen": sighting.last_seen.astimezone(settings.tz).isoformat(),
        "first_seen_label": sighting.first_seen.astimezone(settings.tz).strftime("%H:%M:%S"),
        "last_seen_label": sighting.last_seen.astimezone(settings.tz).strftime("%H:%M:%S"),
        "snapshot": media_path(sighting.snapshot) if sighting.snapshot else None,
    } for sighting, camera_name in rows]


# --------------------------------------------------------------- gallery ----
# Adding corridor faces to the enrolment gallery. Admin-only, because a
# mis-added crop becomes a permanent reference for the wrong person and the
# error compounds silently - see app/services/augment.py.
#
# The scan embeds every capture, which takes seconds, so the result is cached
# in the process and refreshed on demand rather than on every page load.
_AUGMENT_CACHE: dict = {"candidates": None, "scanned_at": 0.0}


def _augment_candidates(refresh: bool = False):
    import time
    from app.services import augment
    if refresh or _AUGMENT_CACHE["candidates"] is None:
        _AUGMENT_CACHE["candidates"] = augment.scan()
        _AUGMENT_CACHE["scanned_at"] = time.time()
    return _AUGMENT_CACHE["candidates"]


def _admin_only(request):
    """The signed-in user if they may reach the infrastructure, else None.

    `app/api/auth.py` already refuses these paths in middleware; this is what
    the handlers use to decide what to RENDER, and the belt to that braces.
    """
    from app.api.auth import current_user
    from app.services import auth as auth_svc
    me = current_user(request)
    return me if auth_svc.can_admin(me) else None


def _can_correct(request):
    """The signed-in user if they may void an event or resolve an unknown.

    Wider than `_admin_only` on purpose: a Davomat operatori may fix the
    record without being handed the cameras, the gallery and the accounts.
    """
    from app.api.auth import current_user
    from app.services import auth as auth_svc
    me = current_user(request)
    return me if auth_svc.can_correct(me) else None


@router.get("/gallery/review", response_class=HTMLResponse)
def gallery_review(request: Request, refresh: str = "", msg: str = "",
                   error: str = ""):
    from app.services import augment
    if not _admin_only(request):
        return HTMLResponse("Admin only", status_code=403)
    cands = [c for c in _augment_candidates(refresh=bool(refresh))
             if not c.rejected]
    groups: dict = {}
    for c in cands:
        groups.setdefault((c.employee_id, c.name), []).append(c)
    # Measured with an empty selection: this reports the ENROLMENT set's own
    # worst pair, which augmentation neither causes nor cures. It used to block
    # every addition; now it is shown for what it is.
    gallery = augment.check_impostors([])
    return render(
        "gallery/review.html", request=request, current_view="gallery:review",
        groups=sorted(groups.items(), key=lambda kv: kv[0][1]),
        total=len(cands), existing=augment.added(), gallery=gallery,
        threshold=settings.threshold_for(settings.recognizer_model),
        live_floor=max(settings.threshold_for(settings.recognizer_model),
                       settings.augment_live_floor),
        # The enrolment set's own worst pairs, with both photographs, so an
        # admin can look at what the warning is actually about and act on it.
        worst_pairs=augment.worst_pairs(),
        scanned_at=_AUGMENT_CACHE["scanned_at"], msg=msg, error=error)


@router.get("/gallery/crop/{key}")
def gallery_crop(request: Request, key: str):
    """Serve one candidate crop.

    `data/debug` is deliberately NOT under the public media mount - it holds
    face images of identified people. So this is admin-gated, and the key is
    resolved through augment.crop_path(), which verifies the resolved file is
    inside the debug directory. A key is caller input; without that check
    "../../etc/passwd" reads whatever the service can.
    """
    from app.services import augment
    if not _admin_only(request):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    p = augment.crop_path(key)
    if p is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="image/jpeg")


@router.get("/gallery/enrolment/{embedding_id}")
def gallery_enrolment(request: Request, embedding_id: int):
    """Serve one enrolment photograph. Admin-gated and containment-checked, for
    the same reasons as /gallery/crop - `face_id_users` holds photographs of
    identified people and the id is caller input."""
    from app.services import augment
    if not _admin_only(request):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    p = augment.enrolment_image(embedding_id)
    if p is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return FileResponse(str(p))


@router.post("/gallery/enrolment/remove")
async def gallery_enrolment_remove(request: Request):
    """Delete enrolment embeddings an admin has judged to be the problem.

    Deliberately separate from the corridor-crop removal, which cannot touch an
    enrolment row at all. This can, so it says plainly what it did and did not
    do - including that the photograph on disk survives and `scripts/enroll.py`
    will bring the row back.
    """
    from app.services import augment
    if not _admin_only(request):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    form = await request.form()
    ids = [int(x) for x in form.getlist("id") if str(x).isdigit()]
    out = await run_in_threadpool(augment.remove_enrolment, ids)
    _AUGMENT_CACHE["candidates"] = None
    try:
        await run_in_threadpool(runtime.reload_gallery)
    except Exception:
        log.exception("gallery reload after enrolment removal failed")
    parts = []
    if out["removed"]:
        parts.append(f"{out['removed']} enrolled photo(s) removed from gallery. "
                     f"The file remains on disk and will be re-added if "
                     f"scripts/enroll.py is run again.")
    parts += out["refused"]
    key = "msg" if out["removed"] else "error"
    return RedirectResponse(
        _p(f"/gallery/review?{key}=" + _quote(" ".join(parts) or "Nothing selected")),
        status_code=303)


@router.post("/gallery/augment")
async def gallery_augment(request: Request):
    """Add the admin's selection - unless it manufactures a false accept."""
    from app.services import augment
    if not _admin_only(request):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    form = await request.form()
    keys = set(form.getlist("key"))
    # A cold cache means augment.scan() embeds every capture - seconds.
    chosen = [c for c in await run_in_threadpool(_augment_candidates)
              if c.key in keys and c.vec is not None and not c.rejected]
    if not chosen:
        return RedirectResponse(_p("/gallery/review?error=Nothing+selected"),
                                status_code=303)

    check = await run_in_threadpool(augment.check_impostors, chosen)
    if not check.safe:
        # Per crop, and it names what each one collides with. The old message
        # reported the gallery's worst pair - two enrolment photographs that no
        # selection could change - and told the admin to deselect one of them.
        listed = "; ".join(f"{name} ({when}) is already reached at {sim:.3f}"
                           for name, when, sim in check.unusable)
        why = (f"Refused {len(check.unusable)} crop(s): another face already "
               f"reaches them at or above the {check.threshold:.3f} floor a "
               f"corridor crop answers to, so they are lookalikes rather than "
               f"references. {listed}. Deselect them; the rest is fine.")
        return RedirectResponse(_p("/gallery/review?error=" + _quote(why)),
                                status_code=303)

    n = await run_in_threadpool(augment.add, chosen)
    _AUGMENT_CACHE["candidates"] = None          # they are in the gallery now
    try:
        await run_in_threadpool(runtime.reload_gallery)
    except Exception:
        log.exception("gallery reload after augment failed")
    floor = max(check.threshold, settings.augment_live_floor)
    msg = (f"Added {n}, each answering to a {floor:.3f} floor rather than the "
           f"{check.threshold:.3f} threshold an enrolment photo gets. The "
           f"highest any other face reaches through them is {check.after:.3f}.")
    return RedirectResponse(_p("/gallery/review?msg=" + _quote(msg)),
                            status_code=303)


@router.post("/gallery/augment/remove")
async def gallery_augment_remove(request: Request):
    from app.services import augment
    if not _admin_only(request):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    form = await request.form()
    ids = [int(x) for x in form.getlist("id") if str(x).isdigit()]
    n = await run_in_threadpool(augment.remove, ids)
    _AUGMENT_CACHE["candidates"] = None
    try:
        await run_in_threadpool(runtime.reload_gallery)
    except Exception:
        log.exception("gallery reload after removal failed")
    return RedirectResponse(_p("/gallery/review?msg=" + _quote(f"Removed {n}.")),
                            status_code=303)


@router.get("/recognition", response_class=HTMLResponse)
def recognition_live(request: Request):
    day = today()
    with session_scope() as s:
        records = _daily_rows(s, day)
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        n_unknown = s.execute(select(func.count(UnknownSighting.id))
                              .where(UnknownSighting.business_date == day)).scalar() or 0
        events = _events(s, 30)
        unknown_attempts = _unknown_activity(s, day, 30)
        cams = s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars().all()

    camera_rows = _live_camera_context(cams)

    checked_out = sum(1 for r in records if r.check_out_time)
    checked_in = sum(1 for r in records if r.check_in_time)
    summary = f"Today {len(records)} of {total} employees were recorded"
    return render(
        "recognition/live.html", request=request, current_view="recognition:live",
        cameras=camera_rows,
        stats={
            "recognized_today": len(records), "total_employees": total,
            "checked_in_today": checked_in, "checked_out_today": checked_out,
            "unknown_attempts": n_unknown,
            "last_event": events[0].timestamp if events else None,
            "summary": summary,
        },
        today_records=records, recent_events=events, unknown_attempts=unknown_attempts,
        attendance_summary=summary,
        recent_events_serialized=_event_payload(events),
        unknown_attempts_serialized=unknown_attempts,
    )


@router.get("/recognition/stream", response_class=HTMLResponse)
def recognition_stream(request: Request):
    return render("recognition/stream.html", request=request, current_view="recognition:live")


@router.get("/recognition/logs")
def recognition_logs():
    """Polled by the live page to refresh its side panels."""
    day = today()
    with session_scope() as s:
        records = _daily_rows(s, day)
        total = s.execute(select(func.count(Employee.id))
                          .where(Employee.is_active.is_(True))).scalar() or 0
        n_unknown = s.execute(select(func.count(UnknownSighting.id))
                              .where(UnknownSighting.business_date == day)).scalar() or 0
        events = _events(s, 30)
        unknown_attempts = _unknown_activity(s, day, 30)
    checked_in = sum(1 for record in records if record.check_in_time)
    checked_out = sum(1 for record in records if record.check_out_time)
    return {
        "employees": [{
            "employee_id": r.employee.employee_id if r.employee else "",
            "name": r.employee.full_name if r.employee else "",
            "department": r.employee.department if r.employee else "",
            "check_in": r.check_in_time.isoformat() if r.check_in_time else None,
            "check_out": r.check_out_time.isoformat() if r.check_out_time else None,
            "check_in_snapshot": r.check_in_snapshot,
            "check_out_snapshot": r.check_out_snapshot,
            "status": ("checked_out" if r.check_out_time
                       else "checked_in" if r.check_in_time else "waiting"),
            "working_hours_seconds": int(r.working_hours.total_seconds()) if r.working_hours else 0,
            "recognition_count": r.recognition_count,
        } for r in records],
        "events": _event_payload(events),
        "unknown_attempts": unknown_attempts,
        "stats": {"recognized_today": len(records), "total_employees": total,
                  "checked_in_today": checked_in, "checked_out_today": checked_out,
                  "unknown_attempts": n_unknown,
                  "summary": f"Today {len(records)} of {total} employees were recorded"},
    }


# /login and /logout now live in app/api/auth.py, which owns the POST handler,
# the session cookie, and the redirect target. A GET-only stub here would
# shadow that router depending on include order.
