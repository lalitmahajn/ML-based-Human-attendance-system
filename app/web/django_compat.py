"""Jinja2 environment that renders the project's original Django templates.

The UI in `templates/` was written for Django.  Rather than rewrite 6,000 lines
of working, well-designed Bootstrap markup, this supplies the handful of Django
constructs it depends on: the `{% load %}` tag, `{% url %}`, and the filters the
templates actually use (`date`, `default`, `duration_hm`, `floatformat`,
`json_script`, `pluralize`, `escapejs`, `yesno`, `add_class`, `media_url`,
`clean_phone`, `length`).

Ported from the previous `fast_api/routers/pages.py`, with the URL map pointed
at the v3 routes and the filters made tolerant of the types the new view models
hand them.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, nodes
from jinja2.ext import Extension
from markupsafe import Markup

from app.config import settings

TEMPLATES_DIR = settings.root / "templates"

# Django |date / |time format strings -> strftime
DJANGO_FORMATS = {
    "H:i": "%H:%M", "H:i:s": "%H:%M:%S", "Y-m-d": "%Y-%m-%d",
    "M d, Y": "%b %d, %Y", "d/m/Y": "%d/%m/%Y", "F j, Y": "%B %d, %Y",
    "d M Y": "%d %b %Y", "M j": "%b %j", "N j, Y": "%b %d, %Y",
    "D, d M Y": "%a, %d %b %Y", "j F Y": "%d %B %Y",
}

URL_MAP = {
    "dashboard:index": "/", "dashboard:home": "/",
    "employees:list": "/employees", "employees:add": "/employees/add",
    "employees:register": "/employees/add", "employees:detail": "/employees/{pk}",
    "employees:edit": "/employees/{pk}/edit", "employees:delete": "/employees/{pk}/delete",
    "recognition:live": "/recognition", "recognition:register": "/employees/add",
    "recognition:logs": "/recognition/logs", "recognition:stream": "/recognition/stream",
    "camera:settings": "/cameras",
    "attendance:list": "/attendance", "attendance:history": "/attendance/history",
    "attendance:unknown_attempts": "/attendance/unknown",
    "attendance:unknown": "/attendance/unknown",
    "attendance:employee_detail": "/attendance/employee/{pk}",
    "attendance:export": "/api/attendance/export",
    "auth:login": "/login", "auth:logout": "/logout", "auth:register": "/register",
}


class LoadExtension(Extension):
    """`{% load ... %}` is a Django no-op here."""
    tags = {"load"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        while parser.stream.current.test("name"):
            next(parser.stream)
        return nodes.Output([nodes.Const("")]).set_lineno(lineno)


def _fmt(value, spec, fallback):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    py = DJANGO_FORMATS.get(spec, spec)
    try:
        return value.strftime(py if "%" in py else fallback)
    except Exception:
        return str(value)


def date_filter(value, spec="Y-m-d"):
    return _fmt(value, spec, "%Y-%m-%d")


def time_filter(value, spec="H:i"):
    return _fmt(value, spec, "%H:%M")


def duration_hm(value):
    """Render a timedelta (or seconds) as HH:MM."""
    if value is None:
        return ""
    try:
        secs = int(value.total_seconds()) if isinstance(value, timedelta) else int(value)
    except Exception:
        return str(value)
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


def default_filter(value, fallback=""):
    return fallback if value is None or value == "" or value == [] else value


def floatformat(value, arg=2):
    try:
        return f"{float(value):.{int(arg)}f}"
    except Exception:
        return value


def pluralize(value, arg="s"):
    try:
        n = int(value if not hasattr(value, "__len__") else len(value))
    except Exception:
        return ""
    if n == 1:
        return ""
    return arg.split(",", 1)[1] if "," in arg else arg


def yesno(value, arg="yes,no,maybe"):
    parts = arg.split(",")
    if value:
        return parts[0]
    if value is None and len(parts) > 2:
        return parts[2]
    return parts[1] if len(parts) > 1 else "no"


# Django's escapejs table. Every character that could end a JS string literal,
# an HTML attribute or a <script> block becomes \uXXXX, which the JS parser
# turns back into the character INSIDE the string. The previous version used
# json.dumps, which leaves ' alone - so a name with an apostrophe closed the
# confirm('...') literal an inline handler had put it in, and whatever followed
# the apostrophe ran as code.
_JS_ESCAPES = {
    ord("\\"): "\\u005C", ord("'"): "\\u0027", ord('"'): "\\u0022",
    ord(">"): "\\u003E", ord("<"): "\\u003C", ord("&"): "\\u0026",
    ord("="): "\\u003D", ord("-"): "\\u002D", ord(";"): "\\u003B",
    ord("`"): "\\u0060", ord("\u2028"): "\\u2028", ord("\u2029"): "\\u2029",
}
_JS_ESCAPES.update((z, "\\u%04X" % z) for z in range(32))

# The same idea for JSON dropped into a <script type="application/json">
# block: json.dumps leaves "</script>" intact, and the HTML parser ends the
# block right there, before the JSON parser ever sees it. \u003C and friends
# are valid JSON and decode to the same characters.
_JSON_SCRIPT_ESCAPES = {ord(">"): "\\u003E", ord("<"): "\\u003C", ord("&"): "\\u0026"}


def escapejs(value):
    return "" if value is None else str(value).translate(_JS_ESCAPES)


def json_script(value, element_id=""):
    body = json.dumps(value, default=str).translate(_JSON_SCRIPT_ESCAPES)
    return Markup(f'<script id="{element_id}" type="application/json">{body}</script>')


def app_prefix() -> str:
    """The deployment's URL prefix, e.g. "/faceid" - or "" at the root."""
    from app.config import settings
    return settings.url_prefix.rstrip("/")


def media_path(value) -> str:
    """Absolute URL for a stored media file, carrying the deployment prefix.

    Under a sub-path deployment the app shares its origin with other projects,
    so a bare "/media/x.jpg" resolves against the DOMAIN root and 404s. Every
    media URL in the app is built here for that reason.
    """
    p = app_prefix()
    v = str(value)
    # IDEMPOTENT. A value that has already been through here - or through
    # `media_url`, which delegates to it - must not be prefixed a second time.
    # Under a sub-path deployment that produced
    # `/faceid/media/faceid/media/snapshots/x.jpg`, which 404s, while the same
    # template rendered correctly on a root deployment because the prefix is
    # empty and the double application is invisible. That is exactly how it
    # reached production: /attendance/unknown showed broken thumbnails whose
    # click-through modal, using the raw value, worked fine.
    if p and v.startswith(f"{p}/"):
        return v
    v = v.lstrip("/")
    if v.startswith("media/"):
        return f"{p}/{v}"
    return f"{p}/media/{v}"


def media_url(value):
    if not value:
        return f"{app_prefix()}/static/img/avatar-placeholder.png"
    v = str(value)
    if v.startswith(("http://", "https://", "data:")):
        return v
    if v.startswith("/static/"):
        return f"{app_prefix()}{v}"
    return media_path(v)


def clean_phone(value):
    return re.sub(r"[^\d+]", "", str(value)) if value else ""


def get_item(d, key):
    if d is None:
        return None
    return d.get(key) if hasattr(d, "get") else getattr(d, key, None)


def add_class(field, css):
    """Templates use this on Django form fields; the v3 pages pass plain values."""
    return field.as_widget(attrs={"class": css}) if hasattr(field, "as_widget") else field


def url(name, *args, **kwargs):
    """Reverse a route name to a path, carrying the deployment prefix.

    The edge proxy forwards /faceid/... unchanged, so a link to "/attendance"
    would leave the mounted app entirely. Every generated link has to include
    the prefix, and this is the one place they are built.
    """
    from app.config import settings
    path = URL_MAP.get(name, f"/{name}")
    pk = kwargs.get("pk") or kwargs.get("id") or (args[0] if args else None)
    if pk is not None:
        path = path.replace("{pk}", str(pk))
    prefix = settings.url_prefix.rstrip("/")
    return f"{prefix}{path}" if prefix else path


class _User:
    """The templates check `user.is_authenticated`; auth is not built yet."""
    is_authenticated = True
    is_staff = True
    is_superuser = True
    username = "admin"


def build_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        extensions=[LoadExtension],
        autoescape=True,
    )
    env.filters.update(
        date=date_filter, time=time_filter, duration_hm=duration_hm,
        default=default_filter, floatformat=floatformat, pluralize=pluralize,
        yesno=yesno, escapejs=escapejs, json_script=json_script,
        media_url=media_url, clean_phone=clean_phone, get_item=get_item,
        add_class=add_class, length=lambda v: len(v) if hasattr(v, "__len__") else 0,
        safe=lambda v: Markup(v), add=lambda v, a: (int(v) + int(a)) if str(v).lstrip("-").isdigit() else v,
    )
    env.globals.update(
        url=url,
        now=lambda: datetime.now(settings.tz),
        today=lambda: datetime.now(settings.tz).date(),
        user=_User(),
        DEBUG=False,
        STATIC_VERSION="3.1.0",
        current_view="",
    )
    return env
