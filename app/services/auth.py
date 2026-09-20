"""Operator accounts: authentication, creation, and the first-run admin.

The console shows attendance movements and face crops of identified people.
Before this existed every page and every JSON endpoint answered anyone who
knew the URL, so `/login` was decoration - the data was one guessed path away.
"""
from __future__ import annotations

import logging
import re

from sqlalchemy import func, select

from app.core.security import hash_password, needs_rehash, verify_password
from app.db.models import User
from app.db.session import session_scope
from app.db.models import utcnow

log = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{3,64}$")
MIN_PASSWORD_LEN = 6

# ---- roles -------------------------------------------------------------
# One table, three roles, and the capability predicates below are the ONLY
# place that answers "may this account do X". Scattering `role == "admin"`
# through the routes is how a page ends up gated and its POST handler not.
ROLE_ADMIN = "admin"
ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"

ROLE_LABELS = {
    ROLE_ADMIN: "Administrator",
    ROLE_OPERATOR: "Attendance Operator",
    ROLE_VIEWER: "Viewer",
}

ROLE_HELP = {
    ROLE_ADMIN: "Full access: cameras, gallery, users, live monitoring.",
    ROLE_OPERATOR: ("Works with attendance and unknowns, can void incorrect "
                    "recognitions. Cameras, gallery, live view, and "
                    "user management are hidden."),
    ROLE_VIEWER: "Read-only access; cannot make any changes.",
}

ROLES = tuple(ROLE_LABELS)


def normalize_role(raw: str | None, *, is_admin: bool = False) -> str:
    """A role name we recognise, or the safest one that fits.

    Unknown input becomes VIEWER rather than raising: this is read from a
    cookie and from a form, and the failure mode of an unrecognised value
    must be less access, never more.
    """
    role = (raw or "").strip().lower()
    if role in ROLE_LABELS:
        return role
    return ROLE_ADMIN if is_admin else ROLE_VIEWER


def session_role(session: dict | None) -> str:
    """The role carried by a signed-in session, or VIEWER for nobody."""
    if not session:
        return ROLE_VIEWER
    return normalize_role(session.get("r"), is_admin=bool(session.get("adm")))


def can_admin(session: dict | None) -> bool:
    """May reach the infrastructure: cameras, live view, gallery, accounts.

    These surfaces show the raw camera feed and can change what the system
    believes a person looks like, which is a strictly larger power than
    correcting a day's attendance.
    """
    return session_role(session) == ROLE_ADMIN


def can_correct(session: dict | None) -> bool:
    """May void a wrong recognition and resolve an unknown sighting.

    Deliberately wider than `can_admin`: an attendance operator is exactly
    the person who knows that the man in the 13:00 crop is not Rustam, and
    making them ask an admin to press the button is how bad rows survive.
    """
    return session_role(session) in (ROLE_ADMIN, ROLE_OPERATOR)


DEFAULT_USERNAME = "inomjon"
DEFAULT_PASSWORD = "123456"


class AuthError(ValueError):
    """Rejected input, safe to show a human."""


def normalize_username(raw: str) -> str:
    return (raw or "").strip().lower()


def validate_new_user(username: str, password: str) -> tuple[str, str]:
    u = normalize_username(username)
    if not USERNAME_RE.match(u):
        raise AuthError(
            "Username must be 3-64 characters, letters, digits, dot, dash or underscore."
        )
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    return u, password


def authenticate(username: str, password: str) -> dict | None:
    """Return a small dict on success, None on any failure.

    One indistinguishable failure for 'no such user', 'wrong password' and
    'account disabled': telling them apart tells an attacker which usernames
    are real. The dummy verify on the no-user path keeps the response time
    flat so absence cannot be timed either.
    """
    u = normalize_username(username)
    with session_scope() as s:
        row = s.execute(select(User).where(User.username == u)).scalar_one_or_none()
        if row is None:
            verify_password(password or "x", _DUMMY_HASH)
            return None
        if not row.is_active or not verify_password(password or "", row.password_hash):
            return None

        # Opportunistic upgrade: if the stored cost is below what we now use,
        # this login is the only moment the plaintext is available to redo it.
        if needs_rehash(row.password_hash):
            row.password_hash = hash_password(password)
        row.last_login_at = utcnow()
        return {"uid": row.id, "username": row.username,
                "is_admin": bool(row.is_admin), "full_name": row.full_name or "",
                "role": normalize_role(row.role, is_admin=bool(row.is_admin))}


def create_user(username: str, password: str, *, is_admin: bool = False,
                full_name: str = "", role: str | None = None) -> dict:
    u, pw = validate_new_user(username, password)
    # `role` wins when given; `is_admin` is what the older callers pass. They
    # are kept in step here so the column and the flag can never disagree -
    # a row claiming is_admin with role "viewer" would authorise differently
    # depending on which one the reader happened to consult.
    role = normalize_role(role, is_admin=is_admin)
    admin = role == ROLE_ADMIN
    with session_scope() as s:
        exists = s.execute(
            select(func.count()).select_from(User).where(User.username == u)
        ).scalar_one()
        if exists:
            raise AuthError(f"User {u!r} already exists.")
        row = User(username=u, password_hash=hash_password(pw),
                   is_admin=admin, role=role,
                   full_name=full_name.strip(), is_active=True)
        s.add(row)
        s.flush()
        log.info("created user %r (role=%s)", u, role)
        return {"uid": row.id, "username": row.username,
                "is_admin": row.is_admin, "role": row.role}


def set_role(username: str, role: str) -> str:
    """Change what an account may do. Refuses to demote the last active admin
    for the same reason `set_active` refuses to disable one."""
    u = normalize_username(username)
    want = (role or "").strip().lower()
    if want not in ROLE_LABELS:
        raise AuthError(f"Unknown role: {role!r}")
    with session_scope() as s:
        row = s.execute(select(User).where(User.username == u)).scalar_one_or_none()
        if row is None:
            raise AuthError(f"No such user: {u}")
        if row.is_admin and want != ROLE_ADMIN:
            others = s.execute(
                select(func.count()).select_from(User).where(
                    User.is_admin.is_(True), User.is_active.is_(True), User.id != row.id)
            ).scalar_one()
            if not others:
                raise AuthError("Cannot demote the last active admin.")
        row.role = want
        row.is_admin = want == ROLE_ADMIN
        log.info("user %r role -> %s", u, want)
        return want


def set_password(username: str, password: str) -> None:
    u = normalize_username(username)
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    with session_scope() as s:
        row = s.execute(select(User).where(User.username == u)).scalar_one_or_none()
        if row is None:
            raise AuthError(f"No such user: {u}")
        row.password_hash = hash_password(password)


def set_active(username: str, active: bool) -> None:
    """Disable rather than delete, so old audit rows keep a name to point at.

    Refuses to disable the last active admin - an unreachable console is worse
    than a stale account.
    """
    u = normalize_username(username)
    with session_scope() as s:
        row = s.execute(select(User).where(User.username == u)).scalar_one_or_none()
        if row is None:
            raise AuthError(f"No such user: {u}")
        if not active and row.is_admin:
            others = s.execute(
                select(func.count()).select_from(User).where(
                    User.is_admin.is_(True), User.is_active.is_(True), User.id != row.id)
            ).scalar_one()
            if not others:
                raise AuthError("Cannot disable the last active admin.")
        row.is_active = bool(active)


def list_users() -> list[dict]:
    with session_scope() as s:
        rows = s.execute(select(User).order_by(User.username)).scalars().all()
        return [{"id": r.id, "username": r.username, "full_name": r.full_name or "",
                 "is_admin": bool(r.is_admin), "is_active": bool(r.is_active),
                 "role": normalize_role(r.role, is_admin=bool(r.is_admin)),
                 "role_label": ROLE_LABELS[normalize_role(
                     r.role, is_admin=bool(r.is_admin))],
                 "created_at": r.created_at, "last_login_at": r.last_login_at}
                for r in rows]


def user_count() -> int:
    with session_scope() as s:
        return s.execute(select(func.count()).select_from(User)).scalar_one()


def ensure_default_admin() -> bool:
    """Create the first admin if the table is empty. Returns True if created.

    Runs at startup so a fresh deployment is reachable without a manual step,
    and does nothing once any account exists - it must never resurrect a
    deleted default or reset a changed password.
    """
    if user_count():
        return False
    create_user(DEFAULT_USERNAME, DEFAULT_PASSWORD, role=ROLE_ADMIN,
                full_name="Inomjon")
    log.warning(
        "created default admin %r with the default password - change it at "
        "/users", DEFAULT_USERNAME)
    return True


# A real hash of a random string, compared against when the username does not
# exist so that path costs the same ~140 ms as a real verify.
_DUMMY_HASH = hash_password("not-a-real-password-placeholder")
