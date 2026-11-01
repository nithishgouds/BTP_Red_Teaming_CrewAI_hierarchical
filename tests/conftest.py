import os
import pathlib

# Ensure LOCALAPPDATA points to a writable workspace folder to avoid Windows permission error in sandbox
_appdata = pathlib.Path(__file__).resolve().parents[1] / ".temp_appdata"
_appdata.mkdir(parents=True, exist_ok=True)
os.environ["LOCALAPPDATA"] = str(_appdata)
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

