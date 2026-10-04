"""Dashboard CSS. The stylesheet itself lives in styles.css next to this file so it can be
edited (and syntax-highlighted) as plain CSS; main.py passes CUSTOM_CSS to Gradio."""
from pathlib import Path

CUSTOM_CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
