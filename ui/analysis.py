"""Analysis run: _analyze generator, sample-data loader and post-analysis reveal."""
import io
import os
from datetime import datetime

import gradio as gr

from app.services import overview_metrics, recurring_issues
from app.services.persistence import compute_file_hash
from app.services.persistence import get_cached_result
from app.services.persistence import save_result
from app.services.pipeline import run_pipeline_from_bytes
from app.services.pipeline import run_pipeline_from_text

from ui.constants import _ANALYZE_N_OUTPUTS, _ANALYZE_STEPS, _EXEC_SUMMARY_PLACEHOLDER, _SAMPLE_DATA_PATH
from ui.helpers import _analysis_to_summary_stats, _results_to_full_dataframe, _truncate_full_df
from ui.charts import _donut_figure, _priority_color, _priority_counts
from ui.components import (
    _agent_progress_html,
    _assignment_group_performance,
    _assignment_group_table_html,
    _bar_list_html,
    _kpi_deltas,
    _loaded_strip_html,
    _overview_kpi_html,
    _recurring_issues_table_html,
    _volume_sparkline_html,
)


def _progress_update(step: int, stage: str):
    return gr.update(value=_agent_progress_html(stage, step, _ANALYZE_STEPS), visible=True)


async def _analyze(file_obj, pasted_text):
    has_file = file_obj is not None
    has_text = bool(pasted_text and pasted_text.strip())

    if not has_file and not has_text:
        raise gr.Error("Upload a file (CSV/XLSX/TXT) or paste incident text first.")
    if has_file and has_text:
        # Both fields are populated - almost always a leftover value from a
        # previous analysis still sitting in the file uploader, not two
        # inputs the user actually intends together. Rather than silently
        # picking the file (which is what used to happen, and produced a
        # confusing "already categorized" result while the newly-pasted
        # text was quietly ignored), make the user disambiguate.
        raise gr.Error(
            "Both a file and pasted text are present - please clear one of them. "
            "(If you analyzed a file earlier, use the ✕ on the file box to remove it "
            "before pasting text, or vice versa.)"
        )

    # Show the animated agent progress bar in its single fixed spot before
    # doing any work; other outputs are left untouched (gr.update()) so
    # nothing under them flickers or shows its own loading state.
    yield (_progress_update(1, "Reading input"),) + (gr.update(),) * (_ANALYZE_N_OUTPUTS - 1)

    # Hash the raw input (file bytes, or the pasted text) so an identical
    # upload/paste can be served straight from the DB instead of hitting
    # the LLM pipeline again.
    if has_file:
        with open(file_obj.name, "rb") as f:
            content = f.read()
        input_label = file_obj.name
        display_label = os.path.basename(file_obj.name)
    else:
        content = pasted_text.strip().encode("utf-8")
        input_label = "pasted_text"
        display_label = "pasted text"

    file_hash = compute_file_hash(content)
    llm_worklog_scoring_enabled = os.environ.get("ENABLE_LLM_WORKLOG_SCORING", "false").lower() == "true"
    cached = get_cached_result(file_hash, current_llm_worklog_scoring_enabled=llm_worklog_scoring_enabled)

    if cached is not None:
        yield (_progress_update(2, "Loading cached results"),) + (gr.update(),) * (_ANALYZE_N_OUTPUTS - 1)
        full_df = cached["full_df"]
        summary_stats = cached["summary_stats"]
        category_counts = cached["category_counts"]
        host_counts = cached["host_counts"]
        cached_on = cached["uploaded_at"][:19].replace("T", " ")
        cache_notice = gr.update(
            value=f"✅ Identical input already analyzed on {cached_on} — showing cached results, no LLM calls made.",
            visible=True,
        )
    else:
        yield (_progress_update(2, "Validating, categorizing & scoring tickets"),) + (gr.update(),) * (_ANALYZE_N_OUTPUTS - 1)
        if has_file:
            analysis = await run_pipeline_from_bytes(file_obj.name, content)
        else:
            analysis = await run_pipeline_from_text(pasted_text)

        full_df = _results_to_full_dataframe(analysis)
        summary_stats = _analysis_to_summary_stats(analysis)
        category_counts = analysis.category_counts
        host_counts = analysis.host_counts

        save_result(
            file_hash=file_hash,
            filename=input_label,
            full_df=full_df,
            summary_stats=summary_stats,
            category_counts=category_counts,
            host_counts=host_counts,
            llm_worklog_scoring_enabled=llm_worklog_scoring_enabled,
        )
        cache_notice = gr.update(value="", visible=False)

    yield (_progress_update(3, "Building dashboards"),) + (gr.update(),) * (_ANALYZE_N_OUTPUTS - 1)

    df = _truncate_full_df(full_df)
    overview_kpis = overview_metrics.compute_overview_kpis(full_df, summary_stats)
    summary = _overview_kpi_html(overview_kpis, _kpi_deltas(full_df), _volume_sparkline_html(full_df))

    # Prepare CSV for download - always the FULL untruncated result set,
    # independent of whatever filter/page/truncation the on-screen table
    # is showing.
    csv_buf = io.StringIO()
    full_df.to_csv(csv_buf, index=False)
    csv_path = f"/tmp/itsm_quality_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w") as f:
        f.write(csv_buf.getvalue())

    # Priority isn't pre-aggregated like category/host counts are, so it's
    # derived here from full_df - works the same for a fresh analysis and
    # a cached result, since both always carry a full_df.
    priority_counts = _priority_counts(full_df)
    chart = _donut_figure(
        priority_counts, "Incidents by Priority", color_fn=_priority_color,
        show_count_in_legend=True, legend_style="dash_incidents",
    )

    category_bar_html = _bar_list_html(
        sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True), multicolor=True
    )
    host_bar_html = _bar_list_html(
        sorted(host_counts.items(), key=lambda kv: kv[1], reverse=True), color="#0f6e7a"
    )
    recurring_issues_html = _recurring_issues_table_html(recurring_issues.detect_exact_recurrence(full_df))
    assignment_group_html = _assignment_group_table_html(_assignment_group_performance(full_df))

    # Aggregate stats handed to the chat/RAG tab alongside the retrieved
    # ticket excerpts - same numbers already shown on the dashboard, just
    # collected into one dict for the chat prompt.
    chat_stats = {
        **summary_stats,
        "category_counts": category_counts,
        "host_counts": host_counts,
    }

    yield (
        gr.update(visible=False), summary, chart, csv_path, df, cache_notice,
        category_bar_html, host_bar_html, recurring_issues_html, assignment_group_html,
        chat_stats,
        # Clear both inputs now that this run is done, so a stale file or
        # stale pasted text can't get silently reused (and mistaken for
        # "both provided") on the next click.
        gr.update(value=None), gr.update(value=""),
        # summary_stats_state - feeds the Overview health/attention/exec
        # summary calculations without needing to re-run the pipeline.
        summary_stats,
        # Overview-tab duplicates of the same category bar-list / priority
        # donut shown on Categorization - identical values, not recomputed.
        category_bar_html, chart,
        # source_label_state - shown in the "Loaded" strip.
        display_label,
    )


async def _analyze_sample():
    """Runs the normal _analyze flow on the bundled sample file (the "Try
    sample data" button). Same outputs/behavior as clicking Analyze."""
    class _SampleFile:
        name = _SAMPLE_DATA_PATH

    async for update in _analyze(_SampleFile(), ""):
        yield update


def _reveal_after_analysis(full_df, source_label):
    """Runs right after _analyze: swaps the first-run hero/upload UI for the
    slim "Loaded" strip, shows the Overview result cards, and unlocks the tabs
    that need analyzed data. Output order matches `_reveal_outputs` in build_ui()."""
    n = 0 if full_df is None else len(full_df)
    tab_update = gr.update(interactive=True) if n > 0 else gr.update()
    return (
        gr.update(visible=False),                                   # hero_html
        gr.update(visible=False),                                   # input_panel
        gr.update(visible=True),                                    # loaded_strip
        _loaded_strip_html(source_label, n),                        # loaded_strip_html
        gr.update(visible=True),                                    # overview_results
        tab_update, tab_update, tab_update, tab_update,             # Categorization, Trends, Recommendations, Q&A
        _EXEC_SUMMARY_PLACEHOLDER,                                  # exec_summary_output (drop a stale summary)
    )
