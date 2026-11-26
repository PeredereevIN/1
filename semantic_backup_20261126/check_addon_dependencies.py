# -*- coding: utf-8 -*-
"""Check exact add-on versions without multiline PowerShell -Command quoting."""
from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
import sys

# YAKE 0.7.3 is the version required by scientific-intelligence-lab 1.0.0
# in the user's environment. UMAP 0.6.0 was present before the previous add-on
# installation changed it. Pin both to avoid a repeat of that environment damage.
REQUIRED = {
    "igraph": ("igraph", "1.0.0"),
    "leidenalg": ("leidenalg", "0.12.0"),
    "yake": ("yake", "0.7.3"),
    "umap": ("umap-learn", "0.6.0"),
    "streamlit": ("streamlit", None),
    "plotly": ("plotly", None),
}

missing: list[str] = []
print("=== Topic discovery dependency check ===")
for module, (package, expected) in REQUIRED.items():
    spec = importlib.util.find_spec(module)
    if spec is None:
        print(f"MISSING: {module} ({package})")
        missing.append(package if expected is None else f"{package}=={expected}")
        continue
    try:
        installed = version(package)
    except PackageNotFoundError:
        installed = "unknown"
    if expected is not None and installed != expected:
        print(f"VERSION MISMATCH: {package}: installed={installed}, required={expected}")
        missing.append(f"{package}=={expected}")
    else:
        print(f"OK: {module} {installed}")
if missing:
    print("Required package repairs: " + ", ".join(missing))
    sys.exit(1)
print("All required dependencies match the add-on compatibility set.")
