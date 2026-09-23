#!/usr/bin/env python3
"""Register the two corridor cameras with their IN/OUT roles."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from app.db.models import Camera, CameraRole
from app.db.session import init_db, session_scope

import os
USER = os.environ.get("CAMERA_USER", "admin")
PWD = os.environ.get("CAMERA_PASSWORD", "admin123")
if "CAMERA_PASSWORD" not in os.environ:
    print("  Notice: CAMERA_PASSWORD not set in env; registered placeholder RTSP credentials (update in /cameras if needed).")
CAMS = [
    ("Entrance", CameraRole.IN,  "192.168.1.2",  "corridor, high mount, wide angle"),
    ("Exit",     CameraRole.OUT, "192.168.1.64", "corridor, high mount, wide angle"),
]

def main():
    init_db()
    with session_scope() as s:
        for name, role, ip, notes in CAMS:
            url = f"rtsp://{USER}:{PWD}@{ip}:554/Streaming/Channels/101"
            cam = s.execute(select(Camera).where(Camera.ip == ip)).scalar_one_or_none()
            if cam is None:
                cam = Camera(ip=ip)
                s.add(cam)
            cam.name, cam.role, cam.rtsp_url, cam.notes, cam.enabled = name, role, url, notes, True
            s.flush()
            print(f"  [{cam.id}] {name:10s} role={role.value:4s} {ip}")
    print("cameras registered")

if __name__ == "__main__":
    main()
