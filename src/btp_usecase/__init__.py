import os
import pathlib
import tempfile

# Ensure LOCALAPPDATA is set to a writable path (prevents CrewAI token manager WinError 5 in sandboxed environments)
_local_app = os.environ.get("LOCALAPPDATA")
try:
    if not _local_app or not os.path.exists(_local_app) or not os.access(_local_app, os.W_OK):
        _fallback = pathlib.Path(tempfile.gettempdir()) / "btp_appdata"
        _fallback.mkdir(parents=True, exist_ok=True)
        os.environ["LOCALAPPDATA"] = str(_fallback)
except Exception:
    pass

def main() -> None:
    print("Hello from btp-usecase!")
