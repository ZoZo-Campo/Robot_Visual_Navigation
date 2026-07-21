"""Streamlit entry point for Robot Visual Navigation V4.

Streamlit reruns this file after every widget interaction. Importing
`ui.streamlit_app` directly would execute it only once because Python keeps
imported modules in `sys.modules`, which can leave the page blank after a
button click. `run_path` executes the UI script on every rerun.
"""

from pathlib import Path
import runpy


APP_DIR = Path(__file__).resolve().parent
runpy.run_path(str(APP_DIR / "ui" / "streamlit_app.py"), run_name="__main__")
