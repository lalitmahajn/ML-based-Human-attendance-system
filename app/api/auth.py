"""Sign-in, sign-out, user administration, and the gate in front of everything.

The gate is middleware rather than a per-route dependency deliberately. There
are ~15 page routes, several JSON endpoints, an MJPEG video feed and a
WebSocket; a dependency has to be remembered on each one, and the failure mode
of forgetting is a silently public endpoint. Middleware is deny-by-default:
a new route is protected the moment it is added, and anything public has to be
named in ONE list below, where it can be reviewed.
"""
from __future__ import annotations

import logging
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.core.security import COOKIE_NAME, MAX_AGE_S, read_session, sign_session
from app.services import auth as auth_svc

log = logging.getLogger(__name__)

router = APIRouter()

# Exact paths served without a session. Everything else requires one.
_PUBLIC_NAMES = ("/login", "/logout", "/favicon.ico", "/health")
_PUBLIC_PREFIX_NAMES = ("/static/",)

# Paths only an ADMIN may reach - the infrastructure of the system rather than
# its output. Same reasoning as the public list above and the same shape: one
# reviewable place, checked in the middleware, so a route added under any of
# these prefixes is restricted the moment it exists rather than whenever
# somebody remembers to decorate it.
#
# Hiding the nav link is not a control. An operator who types /cameras, or
# whose browser replays a bookmark, arrives here.
_ADMIN_PREFIX_NAMES = (
    "/cameras",          # RTSP URLs and camera credentials
    "/gallery",          # changes who the system thinks people are
    "/users",            # accounts
    "/recognition",      # live view and the pipeline's own health
    "/video/",           # the raw camera stream behind the live view
    "/api/health",       # camera-by-camera pipeline stats, same content
    # Enrolment creates an identity the cameras will then trust - it is the
    # gallery by another door, and every signed-in role could reach it.
    "/employees/add",
    "/api/employees",
    "/api/gallery",      # POST /api/gallery/reload
    "/api/debug",        # GET /api/debug/captures: who the debug folder holds
)

# Carved back out of the list above. `users_password` deliberately lets any
# signed-in account change its OWN password (it checks that itself), and a
# prefix match on "/users" would take that away from everyone but admins.
_ADMIN_EXEMPT_NAMES = ("/users/password",)


def _is_admin_path(path: str) -> bool:
    """True for a path in the admin-only list, prefix-resolved.

    Compared against the RAW request path, which under a sub-path deployment
    arrives as /faceid/cameras - so the deployment prefix is applied here for
    the same reason it is in `_is_public`.
    """
    from app.config import settings
    p = settings.url_prefix.rstrip("/")
    if path in {f"{p}{n}" for n in _ADMIN_EXEMPT_NAMES}:
        return False
    return path.startswith(tuple(f"{p}{n}" for n in _ADMIN_PREFIX_NAMES))


def _is_public(path: str) -> bool:
    """Public paths, resolved against the deployment prefix.

    These are compared against the RAW request path, which under a sub-path
    deployment arrives as /faceid/login - so the prefix has to be applied here
    too, or the login page itself would demand a session and loop forever.
    """
    from app.config import settings
    p = settings.url_prefix.rstrip("/")
    if path in {f"{p}{n}" for n in _PUBLIC_NAMES}:
        return True
    return path.startswith(tuple(f"{p}{n}" for n in _PUBLIC_PREFIX_NAMES))


def _p(path: str) -> str:
    """Prefix an absolute app path for the current deployment."""
    from app.config import settings
    return f"{settings.url_prefix.rstrip('/')}{path}"


def current_user(request: Request) -> dict | None:
    return getattr(request.state, "user", None)


def _request_hosts(request: Request) -> set[str]:
    """Hostnames this request may legitimately have been addressed to.

    `request.url.hostname` is the Host header, X-Forwarded-Host is what a
    proxy says the browser asked for, and `trusted_hosts` is what the operator
    declares when neither is the public name. All three are needed because
    none of them is reliably present: see `_cross_site`.
    """
    from app.config import settings
    hosts = {(request.url.hostname or "").lower()}
    for h in request.headers.get("x-forwarded-host", "").split(","):
        h = h.strip().lower()
        if h:
            hosts.add(urlsplit(f"//{h}").hostname or h)
    for h in str(settings.trusted_hosts).split(","):
        h = h.strip().lower()
        if h:
            hosts.add(urlsplit(f"//{h}").hostname or h)
    hosts.discard("")
    return hosts


def _cross_site(request: Request) -> bool:
    """True for a state-changing request that another site sent.

    The session cookie is SameSite=Lax, which already keeps it off cross-site
    form posts in current browsers. This is the second lock, and it reads
    **Sec-Fetch-Site** first: the browser computes that itself, from the page
    that made the request, and sends it on every fetch. Nothing between the
    browser and this process can change it.

    That matters because the obvious test - does the Origin's host match the
    host we were addressed as - is wrong behind a proxy, and this deployment
    has one. The browser posts to `https://aiscan.airi.uz/faceid/users/add`
    with `Origin: https://aiscan.airi.uz`, the proxy forwards it upstream with
    `Host: 127.0.0.1:8081`, and comparing the two refuses the operator's own
    form. Adding a viewer account failed exactly this way.

    So Sec-Fetch-Site decides whenever it is present, which is every browser
    since Chrome 76, Firefox 90 and Safari 16.4. `cross-site` is refused;
    `same-origin`, `same-site` and `none` (typed into the address bar, or a
    bookmark) are allowed. A same-SITE post is allowed deliberately: the
    sibling apps on this host share the origin outright, so no header can tell
    their pages from ours, and pretending otherwise buys nothing.

    Without that header the old host comparison still runs, now including any
    `trusted_hosts` the operator has declared. A request with no Origin and no
    Referer either - curl, a script - is let through: an absent header is not
    evidence of anything, and the cookie check still applies.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return False
    fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
    if fetch_site:
        return fetch_site == "cross-site"
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return False
    # "null" - a sandboxed frame or a privacy redirect - parses to no host and
    # is refused like any other stranger.
    host = (urlsplit(origin).hostname or "").lower()
    return host not in _request_hosts(request)


async def auth_middleware(request: Request, call_next):
    """Attach the signed-in user, or turn the request away.

    HTML requests get a redirect to the login form with the original path in
    `next`; anything expecting JSON gets a 401, because redirecting an XHR to
    an HTML login page produces a confusing parse error rather than a clear
    'you are logged out'.
    """
    path = request.url.path
    if _cross_site(request):
        # Every input to the decision, in one line: a refusal that turns out
        # to be wrong is a locked-out operator, and the whole point of this
        # log is that the next one can be diagnosed without a reproduction.
        log.warning("refused cross-site %s %s (sec-fetch-site %r, origin %r, "
                    "host %r, x-forwarded-host %r, trusted %s)",
                    request.method, path, request.headers.get("sec-fetch-site"),
                    request.headers.get("origin") or request.headers.get("referer"),
                    request.url.hostname, request.headers.get("x-forwarded-host"),
                    sorted(_request_hosts(request)))
        return JSONResponse({"detail": "Cross-site request refused"}, status_code=403)
    session = read_session(request.cookies.get(COOKIE_NAME))
    request.state.user = session

    if session:
        # Signed in, but not necessarily for this. A refusal here is 403 and
        # not a redirect to the login form: they ARE logged in, and bouncing
        # them to a form they would immediately pass tells them nothing.
        if _is_admin_path(path) and not auth_svc.can_admin(session):
            log.warning("role %s denied %s (user %s)",
                        auth_svc.session_role(session), path, session.get("u"))
            return _forbidden(request)
        return await call_next(request)

    if _is_public(path):
        return await call_next(request)

    accept = request.headers.get("accept", "")
    wants_json = (
        path.startswith(_p("/api/"))
        or "application/json" in accept
        or request.headers.get("x-requested-with") == "XMLHttpRequest"
    )
    if wants_json:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    nxt = request.url.path
    if request.url.query:
        nxt = f"{nxt}?{request.url.query}"
    return RedirectResponse(_p(f"/login?next={_quote(nxt)}"), status_code=303)


def _forbidden(request: Request):
    """403, in whichever language the caller speaks."""
    accept = request.headers.get("accept", "")
    if ("application/json" in accept
            or request.headers.get("x-requested-with") == "XMLHttpRequest"
            or request.url.path.startswith(_p("/api/"))):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    from app.api.pages import render
    return HTMLResponse(
        render("auth/forbidden.html", request=request).body, status_code=403)


def _quote(value: str) -> str:
    from urllib.parse import quote
    return quote(value, safe="")


def _safe_next(raw: str | None) -> str:
    """Only absolute paths INSIDE this deployment's prefix.

    Two failures this guards, both seen for real:

    `//evil.example` and scheme-relative forms would bounce a freshly
    authenticated operator to another host.

    Under a sub-path deployment the app shares its origin with unrelated
    projects (aiscan.airi.uz serves /faceid, /manim, /ppe, ...). A bare "/" -
    which is the DEFAULT when the login page is opened directly - is same-site
    and so passed the old check, and sent operators to the domain root instead
    of the console. Anything outside our own prefix is refused for the same
    reason: same-origin is not the same as same-application.
    """
    from app.config import settings
    prefix = settings.url_prefix.rstrip("/")
    # A backslash is refused outright: browsers normalise "/\evil.example"
    # to "//evil.example", so the "//" check alone can be walked around.
    if not raw or not raw.startswith("/") or raw.startswith("//") or "\\" in raw:
        return _p("/")
    if prefix and raw != prefix and not raw.startswith(f"{prefix}/"):
        return _p("/")
    return raw


def _cookie_path() -> str:
    """Where the session cookie applies: this deployment's prefix.

    On the shared host the other projects under the same origin (/manim,
    /ppe, ...) must never receive it; a "/" cookie went to all of them.
    """
    from app.config import settings
    return settings.url_prefix.rstrip("/") or "/"


def _is_https(request: Request | None) -> bool:
    if request is None:
        return False
    if request.url.scheme == "https":
        return True
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return proto == "https"


def _set_session_cookie(response, user: dict, request: Request | None = None) -> None:
    response.set_cookie(
        COOKIE_NAME,
        sign_session(user_id=user["uid"], username=user["username"],
                     is_admin=user["is_admin"], role=user.get("role")),
        max_age=MAX_AGE_S,
        httponly=True,      # not readable from JS, so XSS cannot lift the session
        samesite="lax",     # blocks cross-site form posts, keeps normal links working
        # Secure when the client came in over TLS - directly, or through the
        # edge proxy, which terminates it and says so in X-Forwarded-Proto.
        # Not unconditionally: the LAN deployment is plain http, and a Secure
        # cookie there is silently never sent back, which looks like a login
        # that does not stick.
        secure=_is_https(request),
        path=_cookie_path(),
    )


# Failed sign-ins per client address. Five inside a minute lock that address
# out for a minute: enough to make guessing pointless, short enough that an
# operator who mistyped twice is not on the phone to an admin. In-process,
# so a restart forgets it - acceptable for a limiter whose job is to slow
# things down, not to keep a permanent record.
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_S = 60.0
LOGIN_LOCKOUT_S = 60.0
_LOGIN_FAILURES: dict[str, dict] = {}     # ip -> {"at": [timestamps], "until": lockout end}


def _client_ip(request: Request) -> str:
    """The address the attempt came from.

    Behind the edge proxy every request arrives from the proxy's own address,
    so counting on that would lock the whole site out after five failures by
    anyone. The proxy appends the real client to X-Forwarded-For; the LAST
    entry is the one it wrote itself and the only one a client cannot forge.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        last = fwd.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "?"


def _login_locked_for(ip: str, now: float | None = None) -> float:
    """Seconds left on this address's lockout, or 0."""
    now = time.time() if now is None else now
    entry = _LOGIN_FAILURES.get(ip)
    return max(0.0, entry["until"] - now) if entry else 0.0


def _note_login_failure(ip: str, now: float | None = None) -> None:
    now = time.time() if now is None else now
    # Forget addresses that have gone quiet, or the table only ever grows.
    for quiet in [k for k, v in _LOGIN_FAILURES.items()
                  if v["until"] < now and all(now - t > LOGIN_WINDOW_S for t in v["at"])]:
        del _LOGIN_FAILURES[quiet]
    entry = _LOGIN_FAILURES.setdefault(ip, {"at": [], "until": 0.0})
    entry["at"] = [t for t in entry["at"] if now - t <= LOGIN_WINDOW_S] + [now]
    if len(entry["at"]) >= LOGIN_MAX_FAILURES:
        entry["until"] = now + LOGIN_LOCKOUT_S
        entry["at"] = []


# ---- routes ------------------------------------------------------------

def _render_login(request: Request, *, error: str = "", next_url: str = "/",
                  status: int = 200) -> HTMLResponse:
    from app.api.pages import render
    resp = render("auth/login.html", request=request, current_view="auth:login",
                  error=error, next_url=next_url)
    return HTMLResponse(resp.body, status_code=status)


@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/"):
    if current_user(request):
        return RedirectResponse(_safe_next(next), status_code=303)
    return _render_login(request, next_url=_safe_next(next))


@router.post("/login")
async def login_submit(request: Request,
                       username: str = Form(""), password: str = Form(""),
                       next: str = Form("/")):
    target = _safe_next(next)
    ip = _client_ip(request)
    wait = _login_locked_for(ip)
    if wait > 0:
        # Before the password is even looked at: a locked-out address gets no
        # PBKDF2 work out of us either.
        log.warning("login from %s refused: locked out for %.0fs more", ip, wait)
        resp = _render_login(request, error="Too many failed attempts. Try again in a minute.",
                             next_url=target, status=429)
        resp.headers["Retry-After"] = str(int(wait) + 1)
        return resp
    # PBKDF2 verify is ~140 ms of CPU; off the event loop so one login cannot
    # stall frame delivery to every other viewer.
    from starlette.concurrency import run_in_threadpool
    user = await run_in_threadpool(auth_svc.authenticate, username, password)
    if not user:
        _note_login_failure(ip)
        log.warning("failed login for %r from %s", username, ip)
        return _render_login(request, error="Incorrect username or password.",
                             next_url=target, status=401)
    _LOGIN_FAILURES.pop(ip, None)          # a real sign-in clears the slate
    resp = RedirectResponse(target, status_code=303)
    _set_session_cookie(resp, user, request)
    log.info("login: %s", user["username"])
    return resp


@router.get("/logout")
@router.post("/logout")
def logout():
    resp = RedirectResponse(_p("/login"), status_code=303)
    resp.delete_cookie(COOKIE_NAME, path=_cookie_path())
    if _cookie_path() != "/":
        # Sessions issued before the cookie was scoped to the prefix live at
        # "/". A browser keys cookies on path, so that one has to be named
        # separately or it outlives the sign-out.
        resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@router.get("/users", response_class=HTMLResponse)
def users_page(request: Request, error: str = "", created: str = ""):
    from app.api.pages import render
    me = current_user(request)
    if not auth_svc.can_admin(me):
        return HTMLResponse(
            render("auth/forbidden.html", request=request,
                   current_view="auth:users").body, status_code=403)
    from app.config import settings
    users = auth_svc.list_users()
    for u in users:
        # Stored as UTC; shown in the console's own zone like every other
        # time on every other page.
        if u.get("last_login_at"):
            u["last_login_at"] = u["last_login_at"].astimezone(settings.tz)
    return render("auth/users.html", request=request, current_view="auth:users",
                  users=users, me=me, error=error, created=created,
                  roles=[(r, auth_svc.ROLE_LABELS[r]) for r in auth_svc.ROLES],
                  role_help=auth_svc.ROLE_HELP)


@router.post("/users/add")
def users_add(request: Request, username: str = Form(""), password: str = Form(""),
              full_name: str = Form(""), is_admin: str = Form(""),
              role: str = Form("")):
    me = current_user(request)
    if not auth_svc.can_admin(me):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    try:
        # The form posts a role. `is_admin` is still read for the older
        # checkbox form and for any script that predates roles: absent role
        # plus a non-"0" checkbox still means admin, which is what new
        # accounts defaulted to.
        if role:
            auth_svc.create_user(username, password, role=role, full_name=full_name)
        else:
            admin = str(is_admin).lower() not in {"0", "false", "no"}
            auth_svc.create_user(username, password, is_admin=admin,
                                 full_name=full_name)
    except auth_svc.AuthError as e:
        return RedirectResponse(_p(f"/users?error={_quote(str(e))}"), status_code=303)
    return RedirectResponse(_p(f"/users?created={_quote(username.strip().lower())}"),
                            status_code=303)


@router.post("/users/toggle")
def users_toggle(request: Request, username: str = Form(""), active: str = Form("1")):
    me = current_user(request)
    if not auth_svc.can_admin(me):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    try:
        auth_svc.set_active(username, str(active) not in {"0", "false", "no"})
    except auth_svc.AuthError as e:
        return RedirectResponse(_p(f"/users?error={_quote(str(e))}"), status_code=303)
    return RedirectResponse(_p("/users"), status_code=303)


@router.post("/users/role")
def users_role(request: Request, username: str = Form(""), role: str = Form("")):
    """Change what an account may do.

    Nothing re-issues the target's cookie, so a role change lands on their
    next sign-in - or within MAX_AGE_S, whichever comes first. That is a
    deliberate limit and not a bug to work around silently: it is stated on
    the page so an admin who has just removed somebody's access knows the
    old session is still good until it expires.
    """
    me = current_user(request)
    if not auth_svc.can_admin(me):
        return JSONResponse({"detail": "Admin only"}, status_code=403)
    try:
        applied = auth_svc.set_role(username, role)
    except auth_svc.AuthError as e:
        return RedirectResponse(_p(f"/users?error={_quote(str(e))}"), status_code=303)
    label = auth_svc.ROLE_LABELS[applied]
    note = (f"{auth_svc.normalize_username(username)} is now {label}. "
            f"The change will take effect upon their next login.")
    return RedirectResponse(_p(f"/users?created={_quote(note)}"), status_code=303)


@router.post("/users/password")
def users_password(request: Request, username: str = Form(""), password: str = Form("")):
    me = current_user(request)
    if not me:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    target = auth_svc.normalize_username(username)
    # An admin may reset anyone; everyone else only themselves.
    if not auth_svc.can_admin(me) and target != me.get("u"):
        return JSONResponse({"detail": "You may only change your own password"},
                            status_code=403)
    try:
        auth_svc.set_password(target, password)
    except auth_svc.AuthError as e:
        return RedirectResponse(_p(f"/users?error={_quote(str(e))}"), status_code=303)
    return RedirectResponse(_p("/users?created=password-changed"), status_code=303)
