"""Dashboard CSS and Gradio theme. The stylesheet itself lives in styles.css next to this file so it
can be edited (and syntax-highlighted) as plain CSS; main.py passes CUSTOM_CSS and ITSM_THEME to Gradio.

ITSM_THEME gives Gradio's own widgets (inputs, dropdowns, tables, chat) the same teal primary and
Public Sans font as the dashboard CSS, in both light and dark mode."""
from pathlib import Path

import gradio as gr

CUSTOM_CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")

_TEAL = gr.themes.Color(
    c50="#e8f1f3", c100="#d1e5e8", c200="#a3cbd1", c300="#75b1ba", c400="#479aa6",
    c500="#1f8391", c600="#0f6e7a", c700="#0b5862", c800="#08434b", c900="#052e34", c950="#031d21",
    name="itsm_teal",
)

ITSM_THEME = gr.themes.Soft(
    primary_hue=_TEAL,
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Public Sans"), "Inter", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
)
