"""
Gradio dashboard for the ITSM Quality Analysis Agent - entry point.

Runs in-process (mounted into the FastAPI app in main.py) so it calls the pipeline directly
rather than round-tripping through HTTP. The REST API (/api/v1/...) remains available
separately for automation, secured with its own API key.

The implementation is split across this package:

    styles.py / styles.css   dashboard CSS
    constants.py             static text, HTML snippets, column lists, palettes
    helpers.py               formatting + DataFrame preparation
    charts.py                Plotly figures
    components.py            HTML renderers (KPI cards, bar lists, tables, ...)
    llm.py                   LLM provider chain for the Generate summary buttons
    analysis.py              the _analyze run + post-analysis reveal
    tab_*.py                 per-tab refresh/generate handlers
    layout.py                build_ui(): layout and event wiring

main.py keeps importing `CUSTOM_CSS` and `build_ui` from here, unchanged.
"""
from ui.layout import build_ui
from ui.styles import CUSTOM_CSS

__all__ = ["CUSTOM_CSS", "build_ui"]
