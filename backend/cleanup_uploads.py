"""24-hour auto-delete of uploaded photos (privacy — docs/privacy.md).

Runs (a) inside the app before every upload and on startup, and (b) as a
standalone script for a scheduled job: `python backend/cleanup_uploads.py`.
"""
import os
import shutil
import time

from config import UPLOADS_DIR, UPLOAD_TTL_HOURS


def purge_expired_uploads() -> int:
    """Delete session upload folders older than the TTL. Returns count."""
    if not os.path.isdir(UPLOADS_DIR):
        return 0
    cutoff = time.time() - UPLOAD_TTL_HOURS * 3600
    removed = 0
    for name in os.listdir(UPLOADS_DIR):
        path = os.path.join(UPLOADS_DIR, name)
        if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


if __name__ == "__main__":
    print(f"purged {purge_expired_uploads()} expired upload folder(s)")
