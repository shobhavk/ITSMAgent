"""Recommendations tab: card refresh and the management write-up handler."""
import gradio as gr
import pandas as pd

from app.services import recommendations

from ui.constants import _RECOMMENDATIONS_PLACEHOLDER
from ui.components import _recommendations_html
from ui.llm import _llm_recommendations_writeup


def _refresh_recommendations(full_df: pd.DataFrame):
    """Recomputes the Recommendations panel from the latest analysis.
    Deterministic and LLM-free, so it can run automatically at the end of
    every analysis without adding latency or an API dependency - the
    panel is always populated the moment an analysis finishes.

    Returns (cards_html, recommendations_state) so the LLM write-up
    button can reuse the already-computed result instead of recomputing
    it."""
    try:
        result = recommendations.generate_recommendations(full_df)
        return _recommendations_html(result), result
    except Exception:
        return _RECOMMENDATIONS_PLACEHOLDER, {}


async def _generate_recommendations_writeup(full_df: pd.DataFrame, cached_result: dict):
    """Click handler for the Recommendations tab's write-up button.
    Reuses the recommendations already computed when the analysis
    finished (cached_result) rather than recalculating them, so the
    narrative can never describe different numbers than the cards above
    it. Only the small aggregated payload reaches the LLM."""
    try:
        if full_df is None or len(full_df) == 0:
            return gr.update(value="Run an analysis on the Overview tab first, then generate the write-up.")

        result = cached_result if (cached_result or {}).get("available") else \
            recommendations.generate_recommendations(full_df)

        if not result.get("available") or not result.get("recommendations"):
            return gr.update(value=recommendations.fallback_recommendations_markdown(result))

        payload = recommendations.build_llm_payload(result)
        text, used_llm = await _llm_recommendations_writeup(payload)
        if not used_llm:
            return gr.update(
                value=recommendations.fallback_recommendations_markdown(result)
                + "\n\n_(LLM unavailable right now - showing the rule-based write-up of the same recommendations.)_"
            )
        return gr.update(value=text)
    except Exception as exc:
        return gr.update(value=f"Could not generate the recommendations write-up right now ({exc}).")
