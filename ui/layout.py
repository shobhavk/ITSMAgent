"""Gradio layout (build_ui) and event wiring."""
import os

import gradio as gr
import pandas as pd

from ui.constants import (
    DEFAULT_PAGE_SIZE,
    HERO_HTML,
    SEVERITY_NOTE,
    SIDEBAR_BRAND_HTML,
    TOPBAR_HTML,
    _ATTENTION_PLACEHOLDER,
    _EXEC_SUMMARY_PLACEHOLDER,
    _HEALTH_PLACEHOLDER,
    _RECOMMENDATIONS_PLACEHOLDER,
    _SAMPLE_DATA_PATH,
    _TIMELINE_PLACEHOLDER_MD,
)
from ui.charts import _refresh_category_heatmap
from ui.components import (
    AGENT_PROGRESS_HTML,
    _OVERVIEW_KPI_PLACEHOLDER,
    _RESOLUTION_METRICS_PLACEHOLDER,
    _timeline_kpi_html,
)
from ui.tab_overview import _generate_executive_summary, _refresh_overview
from ui.tab_trends import (
    _detect_semantic_recurrence,
    _generate_timeline_summary,
    _refresh_incident_timeline,
    _refresh_trend,
)
from ui.tab_recommendations import _generate_recommendations_writeup, _refresh_recommendations
from ui.styles import CUSTOM_CSS
from ui.tab_results import (
    _CAT_SUMMARY_PLACEHOLDER,
    _change_page_size,
    _close_row_detail,
    _filter_view,
    _go_to_page,
    _refresh_categorization_controls,
    _refresh_view,
    _show_row_detail,
)
from ui.tab_chat import _build_chat_index, _chat_clear, _chat_respond
from ui.tab_kb import _kb_delete, _kb_refresh, _kb_reindex, _kb_upload
from ui.analysis import _analyze, _analyze_sample, _reveal_after_analysis


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ITSM Quality Analysis Agent", css=CUSTOM_CSS) as demo:
        # App shell (sidebar removed): a slim dark brand bar spans the full
        # width, followed by a single light content column - navigation is
        # now the icon tab bar built into gr.Tabs below (native Gradio
        # tabs, previously hidden and driven by sidebar buttons instead).
        gr.HTML(SIDEBAR_BRAND_HTML, elem_id="topbar-wrap")

        with gr.Column(elem_id="main-col"):
            gr.HTML(TOPBAR_HTML, elem_id="topbar")
            gr.Markdown(f"_{SEVERITY_NOTE}_", elem_classes=["severity-note"])

            with gr.Tabs(elem_id="main-tabs") as main_tabs:
                with gr.Tab("🏠 Overview", id=0):
                    agent_progress = gr.HTML(AGENT_PROGRESS_HTML, elem_id="agent-progress", visible=False)
                    cache_notice = gr.Markdown(visible=False, elem_id="cache-notice")

                    # First-run hero: shown until the first analysis completes.
                    hero_html = gr.HTML(HERO_HTML, elem_id="hero")

                    # Upload / paste / Analyze bar, now the first thing on the
                    # tab. Wrapped in a Column so it can be collapsed as one
                    # unit after an analysis and re-opened with the button in
                    # the "Loaded" strip below.
                    with gr.Column(elem_id="input-panel") as input_panel:
                        # Single full-width input bar - upload, paste, and the
                        # analyze action sit on one row (download lives under
                        # the Recent Incidents table in Categorization). Now at
                        # the top of the tab; collapses into the "Loaded" strip
                        # after an analysis (see _reveal_after_analysis).
                        with gr.Row(elem_id="input-row", elem_classes=["dash-card"], equal_height=False):
                            with gr.Column(scale=3, min_width=260):
                                file_input = gr.File(
                                    label="Upload incident file (.xlsx, .csv, .txt)",
                                    file_types=[".xlsx", ".xls", ".csv", ".txt"],
                                    elem_id="file-upload",
                                )
                            with gr.Column(scale=4, min_width=320):
                                text_input = gr.Textbox(label="...or paste unstructured incident text", lines=2,
                                                          placeholder="INC0012345\nShort description: ...\nWorklog: ...")
                            with gr.Column(scale=2, min_width=180, elem_id="action-col"):
                                analyze_btn = gr.Button("Analyze", variant="primary")
                                sample_btn = gr.Button(
                                    "✨ Try sample data", size="sm", elem_id="sample-btn",
                                    visible=os.path.exists(_SAMPLE_DATA_PATH),
                                )

                    # Slim status strip shown after an analysis in place of the
                    # hero + input bar.
                    with gr.Row(elem_id="loaded-strip", elem_classes=["dash-card"], visible=False, equal_height=False) as loaded_strip:
                        loaded_strip_html = gr.HTML("")
                        new_analysis_btn = gr.Button("＋ Analyze new data", size="sm", scale=0, min_width=170)

                    # Everything that is empty before the first analysis lives
                    # in this wrapper, which stays hidden until results exist
                    # (revealed by _reveal_after_analysis). The component
                    # wiring inside is unchanged.
                    with gr.Column(visible=False, elem_id="overview-results") as overview_results:
                        # KPI strip - headline numbers for management at a glance,
                        # shown once an analysis has run (hidden before that - see
                        # overview_results).
                        with gr.Row(elem_id="metrics-row"):
                            summary_md = gr.HTML(_OVERVIEW_KPI_PLACEHOLDER)

                        # (Now sits directly below the KPI strip, so the write-up reads against
                        # the numbers it summarizes.)
                        # Executive Summary - moved to sit right below the page
                        # heading (top bar above the tabs) rather than at the
                        # bottom of the tab, so the management write-up is the
                        # first thing seen. Computes KPIs/trends with pandas
                        # first, then sends only that small aggregated dict to
                        # the LLM for the write-up. The button is NOT inside a
                        # gr.Row with the heading - Gradio gives Row children
                        # negative side margins for edge-to-edge layout, which
                        # was pushing the button past the card's own border.
                        # Instead it's a normal sibling, pinned on top of the
                        # card with CSS position:absolute, so it can never
                        # escape the card's visible edges.
                        with gr.Column(elem_classes=["dash-card"], elem_id="exec-summary-card"):
                            gr.Markdown("### 🧾 Executive Summary", elem_classes=["section-heading"])
                            exec_summary_btn = gr.Button(
                                "✨ Generate summary", size="sm", elem_id="exec-summary-btn",
                            )
                            exec_summary_output = gr.Markdown(
                                _EXEC_SUMMARY_PLACEHOLDER,
                                elem_id="exec-summary-output",
                            )

                        # Incidents by Category / Priority - the same two
                        # panels shown on the Categorization tab, copied
                        # here so management sees the breakdown without
                        # switching tabs. Separate component instances
                        # (Gradio can't render one component in two
                        # places), both populated from the exact same
                        # values _analyze already computes for the
                        # Categorization tab's copies - see category_bar_html /
                        # category_chart in the outputs list below.
                        with gr.Row(elem_id="overview-panel-row-1"):
                            with gr.Column(scale=1, elem_classes=["dash-card"]):
                                gr.Markdown("### 🗂️ Incidents by Category", elem_classes=["section-heading"])
                                overview_category_bar_html = gr.HTML(
                                    '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                                )
                            with gr.Column(scale=1, elem_classes=["dash-card"]):
                                gr.Markdown("### 🎯 Incidents by Priority", elem_classes=["section-heading"])
                                overview_priority_chart = gr.Plot(show_label=False)

                        # Incident Health - four traffic-light indicators so
                        # management can scan overall status in a second.
                        with gr.Column(elem_classes=["dash-card"]):
                            gr.Markdown("### 🩺 Incident Health", elem_classes=["section-heading"])
                            health_html = gr.HTML(_HEALTH_PLACEHOLDER)

                    # Incident Volume Trend - hidden per request, but the
                    # component and _refresh_overview's wiring are left
                    # completely untouched (still receives its value every
                    # refresh) so nothing downstream has to change - only
                    # its visibility here.
                    with gr.Column(elem_classes=["dash-card"], visible=False):
                        gr.Markdown("### 📈 Incident Volume Trend", elem_classes=["section-heading"])
                        overview_trend_chart = gr.Plot(show_label=False)

                    # Attention Required - hidden per request, same
                    # approach as Incident Volume Trend above: the
                    # component still exists and still updates, it's just
                    # not shown.
                    with gr.Column(elem_classes=["dash-card"], visible=False):
                        gr.Markdown("### 🚩 Attention Required", elem_classes=["section-heading"])
                        attention_html = gr.HTML(_ATTENTION_PLACEHOLDER)

                with gr.Tab("🗂️ Categorization", id=1, interactive=False) as tab_categorization:
                    # Categorization tab, laid out as: summary strip -> incident
                    # table (with filter bar) -> Category x Priority chart + top
                    # categories. Clicking a table row opens a detail drawer.
                    # category_chart (the priority donut) is kept in a hidden
                    # column so _analyze's output contract stays untouched.
                    cat_summary_html = gr.HTML(_CAT_SUMMARY_PLACEHOLDER, elem_id="cat-summary")

                    # Recent Incidents table with its original wiring
                    # (_refresh_view / prev_btn / next_btn / page size). The
                    # filter bar below drives it through _filter_view.
                    #
                    # download_file: icon-only gr.DownloadButton pinned into
                    # this card's heading corner (see #results-download-btn in
                    # CUSTOM_CSS); it still receives the same full CSV path
                    # from _analyze.
                    with gr.Column(elem_id="results-section", elem_classes=["dash-card"]):
                        gr.Markdown("### 📋 Recent Incidents (Analyzed &amp; Categorized)", elem_classes=["section-heading"])
                        gr.Markdown(
                            "Includes each incident's timeline: when it came in, time to acknowledge, any "
                            "reassignment found in External Info, and whether External Info carries a "
                            "timestamp trail (best-effort read of the text). Click a row for full details; "
                            "scroll right for all columns.",
                            elem_classes=["severity-note"],
                        )
                        download_file = gr.DownloadButton(
                            "⬇", elem_id="results-download-btn", size="sm",
                        )
                        with gr.Row(elem_id="cat-filter-bar", equal_height=False):
                            search_box = gr.Textbox(
                                placeholder="Search ticket ID, description or worklog…",
                                show_label=False, container=False, scale=4, min_width=220,
                                elem_id="cat-search",
                            )
                            category_filter = gr.Dropdown(
                                choices=["All"], value="All", label="Category",
                                scale=2, min_width=170, elem_id="cat-filter-category",
                            )
                            priority_filter = gr.Dropdown(
                                choices=["All"], value="All", label="Priority",
                                scale=1, min_width=120, elem_id="cat-filter-priority",
                            )
                            review_only = gr.Checkbox(
                                label="Needs review only", value=False,
                                scale=1, min_width=210, elem_id="cat-review-only",
                            )
                        results_table = gr.Dataframe(
                            label=None,
                            show_label=False,
                            interactive=False,
                            wrap=False,
                            max_height=460,
                            pinned_columns=1,
                            elem_id="results-table",
                        )
                        with gr.Row(elem_id="pagination-row"):
                            prev_btn = gr.Button("← Previous", size="sm")
                            page_indicator = gr.Markdown("Page 1 of 1  ·  0 tickets", elem_id="page-indicator")
                            page_size_dropdown = gr.Dropdown(
                                choices=[10, 25, 50, 100], value=DEFAULT_PAGE_SIZE,
                                label="Rows/page", show_label=True,
                                scale=0, min_width=130, elem_id="page-size-dropdown",
                            )
                            next_btn = gr.Button("Next →", size="sm")

                    with gr.Row(elem_id="cat-analytics-row", equal_height=False):
                        with gr.Column(scale=2, elem_classes=["dash-card"]):
                            gr.Markdown("### 📊 Category × Priority", elem_classes=["section-heading"])
                            category_heatmap = gr.Plot(show_label=False)
                        with gr.Column(scale=1, elem_classes=["dash-card"]):
                            gr.Markdown("### 🗂️ Top Categories", elem_classes=["section-heading"])
                            category_bar_html = gr.HTML(
                                '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                            )
                    with gr.Column(visible=False):
                        category_chart = gr.Plot(show_label=False)

                    # Row detail drawer (opened by clicking a table row).
                    with gr.Column(elem_id="detail-drawer", visible=False) as detail_drawer:
                        detail_close_btn = gr.Button("✕ Close", size="sm", elem_id="detail-close-btn")
                        row_detail_html = gr.HTML("", elem_id="row-detail")

                with gr.Tab("📊 Trends & Insights", id=2, interactive=False) as tab_trends:
                    # KPI trend - ticket volume + worklog quality over
                    # time, toggle between daily/weekly/monthly views.
                    with gr.Column(elem_classes=["dash-card"]):
                        gr.Markdown("### 📈 Ticket Trend", elem_classes=["section-heading"])
                        trend_granularity = gr.Radio(
                            ["Daily", "Weekly", "Monthly"], value="Daily",
                            show_label=False, elem_id="trend-granularity",
                        )
                        trend_chart = gr.Plot(show_label=False)

                    # Resolution metrics - MTTD/MTTA/MTTR/SLA. Hidden per
                    # request; the component and its wiring are untouched
                    # (still receives its value every refresh), only its
                    # visibility here changed.
                    with gr.Column(elem_classes=["dash-card"], visible=False):
                        gr.Markdown("### ⏱️ Resolution Metrics", elem_classes=["section-heading"])
                        resolution_metrics_html = gr.HTML(_RESOLUTION_METRICS_PLACEHOLDER)

                    # The standalone "Incident Timeline & External Info Audit"
                    # card was removed from this tab - its per-incident data
                    # now lives in the Recent Incidents table (Categorization
                    # tab). These components stay in a hidden column only so
                    # the existing wiring (_refresh_incident_timeline etc.)
                    # keeps working untouched.
                    with gr.Column(visible=False):
                        timeline_kpi_html = gr.HTML(_timeline_kpi_html({}))
                        timeline_table = gr.Dataframe(
                            label=None, show_label=False, interactive=False,
                            wrap=False, max_height=360, elem_id="timeline-table",
                        )
                        timeline_summary_btn = gr.Button("✨ Generate summary", size="sm")
                        timeline_summary_output = gr.Markdown(_TIMELINE_PLACEHOLDER_MD)

                    # Recurring issues, exact-match: (host, category)
                    # combos meeting a recurrence threshold, with a
                    # real time dimension (first/last seen, average
                    # days between occurrences) - not just a raw count.
                    with gr.Column(elem_classes=["dash-card"]):
                        gr.Markdown("### 🔁 Recurring Issues", elem_classes=["section-heading"])
                        recurring_issues_html = gr.HTML(
                            '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                        )

                    # Recurring issues, semantic: catches the same
                    # underlying problem even when it's logged under
                    # different categories or worded differently each
                    # time. Opt-in (button) rather than automatic,
                    # since clustering needs the whole batch embedded
                    # first - one embedding call, same cost the chat
                    # tab pays lazily on its first question. Reuses
                    # chat_index_state so whichever feature runs first
                    # makes the other free.
                    with gr.Column(elem_classes=["dash-card"]):
                        gr.Markdown(
                            "### 🔬 Semantic Recurrence Detection",
                            elem_classes=["section-heading"],
                        )
                        gr.Markdown(
                            "Finds recurring issues that don't share an exact category or host - "
                            "e.g. the same underlying problem logged inconsistently. Costs one "
                            "embedding call for the batch the first time it (or the chat tab) runs.",
                            elem_classes=["severity-note"],
                        )
                        semantic_recurrence_btn = gr.Button("Detect Similar Recurring Issues", size="sm")
                        semantic_recurrence_html = gr.HTML(
                            '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis, then click the button above.</p>'
                        )

                    # Top hosts, assignment group performance.
                    with gr.Row(elem_id="panel-row-2"):
                        with gr.Column(scale=1, elem_classes=["dash-card"]):
                            gr.Markdown("### 🖥️ Top Affected Servers / Hosts", elem_classes=["section-heading"])
                            host_bar_html = gr.HTML(
                                '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                            )
                        with gr.Column(scale=1, elem_classes=["dash-card"]):
                            gr.Markdown("### 👥 Assignment Group Performance", elem_classes=["section-heading"])
                            assignment_group_html = gr.HTML(
                                '<p style="color:var(--dash-text-muted); font-size:0.85rem; margin:0;">Run an analysis to see this.</p>'
                            )

                with gr.Tab("💡 Recommendations", id=3, interactive=False) as tab_recommendations:
                    gr.Markdown(
                        "Data-backed recommendations for this batch. Every figure below is "
                        "calculated with pandas from the incidents you analyzed - recurrence, "
                        "priority mix, SLA attainment, per-group resolution times, worklog "
                        "quality, and volume concentration. Nothing here is generic advice, and "
                        "an area that crosses no threshold simply isn't listed.",
                        elem_classes=["severity-note"],
                    )

                    # Optional LLM layer: re-voices the cards below for a
                    # management audience. Only the small aggregated
                    # payload is sent (see _llm_recommendations_writeup);
                    # the cards themselves never depend on it. Placed
                    # above Recommended Actions per request.
                    with gr.Column(elem_classes=["dash-card"], elem_id="rec-writeup-card"):
                        gr.Markdown("### 🧾 Management Write-Up", elem_classes=["section-heading"])
                        rec_writeup_btn = gr.Button(
                            "✨ Generate write-up", size="sm", elem_id="rec-writeup-btn",
                        )
                        rec_writeup_output = gr.Markdown(
                            "Run an analysis, then click **Generate write-up** to turn the "
                            "recommendations below into a management-ready briefing. The same "
                            "numbers are used either way.",
                            elem_id="rec-writeup-output",
                        )

                    with gr.Column(elem_classes=["dash-card"]):
                        gr.Markdown("### 💡 Recommended Actions", elem_classes=["section-heading"])
                        recommendations_html = gr.HTML(_RECOMMENDATIONS_PLACEHOLDER)

                with gr.Tab("📚 Knowledge Base", id=4):
                    gr.Markdown(
                        "Upload organizational documents (runbooks, SOPs, troubleshooting "
                        "guides, known-error documents, escalation procedures) so the Agent "
                        "can answer documentation questions - e.g. *\"what does the runbook "
                        "recommend for database connection errors?\"* - separately from the "
                        "incident-data questions on the Q&A tab. Only the relevant retrieved "
                        "passages are ever sent to the model, never the whole document.",
                        elem_classes=["severity-note"],
                    )

                    with gr.Column(elem_classes=["dash-card"]):
                        gr.Markdown("### 📤 Upload Document", elem_classes=["section-heading"])
                        with gr.Row():
                            kb_file_input = gr.File(
                                label="Supported: PDF, DOCX, TXT, MD",
                                file_types=[".pdf", ".docx", ".txt", ".md"],
                                scale=3,
                            )
                            kb_upload_btn = gr.Button("Upload & Index", variant="primary", scale=1)
                        kb_upload_status = gr.Markdown("")

                    with gr.Column(elem_classes=["dash-card"]):
                        gr.Markdown("### 📚 Indexed Documents", elem_classes=["section-heading"])
                        kb_documents_table = gr.Dataframe(
                            headers=["Document Name", "Type", "Status", "Chunks", "Uploaded At", "Error"],
                            interactive=False, wrap=True, elem_id="kb-documents-table",
                        )
                        with gr.Row():
                            kb_document_dropdown = gr.Dropdown(
                                label="Select a document to manage", choices=[], scale=3,
                            )
                            kb_reindex_btn = gr.Button("🔁 Reindex", scale=1)
                            kb_delete_btn = gr.Button("🗑️ Delete", scale=1, variant="stop")
                        kb_manage_status = gr.Markdown("")

                with gr.Tab("💬 Q&A (Agent)", id=5, interactive=False) as tab_qa:
                    # Single bounded chat panel (intro + transcript +
                    # composer) instead of loosely stacked components -
                    # keeps the tab a fixed height with the transcript
                    # scrolling internally like a normal chat app.
                    with gr.Column(elem_id="chat-panel", elem_classes=["dash-card"]):
                        gr.Markdown(
                            "Ask a question about the tickets you just analyzed - e.g. "
                            "*\"what's driving high-priority incidents?\"*, "
                            "*\"which assignment group has the worst worklog quality?\"*, or "
                            "*\"summarize the recurring issues on our database servers.\"* "
                            "Answers are grounded only in the analyzed batch (Overview tab) - "
                            "run an analysis first if you haven't yet. You can also ask "
                            "documentation questions, e.g. *\"what does the runbook recommend "
                            "for connection errors?\"*, answered from the Knowledge Base tab.",
                            elem_classes=["severity-note", "chat-intro"],
                        )
                        chatbot = gr.Chatbot(height=440, show_label=False, elem_id="chatbot")
                        with gr.Row(elem_id="chat-input-row"):
                            chat_input = gr.Textbox(
                                placeholder="Ask a question about the analyzed tickets...",
                                show_label=False, scale=5, container=False,
                            )
                            chat_send = gr.Button("Send", variant="primary", scale=1)
                        chat_clear_btn = gr.Button("Clear conversation", size="sm", elem_id="chat-clear-btn")

                with gr.Tab("⚙️ Settings", id=6):
                    with gr.Column(elem_classes=["dash-card"]):
                        gr.Markdown("### ⚙️ Settings", elem_classes=["section-heading"])
                        gr.Markdown(
                            "Nothing configurable here yet - this tab is a placeholder for "
                            "future options (e.g. default page size, scoring thresholds).",
                            elem_classes=["severity-note"],
                        )

        full_results_state = gr.State(pd.DataFrame())
        # Display label of the last analyzed input (file name / "pasted text"),
        # shown in the "Loaded" strip.
        source_label_state = gr.State("")
        filtered_results_state = gr.State(pd.DataFrame())
        page_state = gr.State(1)
        # Category/score filters still have no UI control (category_state/
        # min_score_state stay fixed at "All"/0, keeping _apply_filters'
        # existing behavior unchanged). Rows-per-page now has one -
        # page_size_dropdown above drives page_size_state directly.
        category_state = gr.State("All")
        min_score_state = gr.State(0)
        page_size_state = gr.State(DEFAULT_PAGE_SIZE)

        # Chat/RAG state: the semantic index + aggregate stats are rebuilt
        # from the latest analysis; chat_history_state is the running
        # (question, answer) transcript fed back into the prompt for
        # multi-turn context.
        chat_index_state = gr.State(None)
        chat_stats_state = gr.State({})
        chat_history_state = gr.State([])

        # Raw summary_stats dict (total/valid/rejected records, avg
        # worklog score) from the latest analysis - feeds the Overview
        # KPI/health/attention calculations and the executive summary
        # without needing to re-run the pipeline.
        summary_stats_state = gr.State({})
        # Holds the recommendations computed at the end of each analysis,
        # so the Management Write-Up button re-voices exactly those rather
        # than recomputing (and possibly drifting from) the cards on screen.
        recommendations_state = gr.State({})
        # Holds the timeline aggregate computed at the end of each
        # analysis, so the "Generate summary" button re-voices exactly
        # those numbers rather than recomputing (and possibly drifting
        # from) the table/KPI cards on screen.
        timeline_aggregate_state = gr.State({})

        _analyze_outputs = [
            agent_progress, summary_md, category_chart, download_file,
            full_results_state, cache_notice,
            category_bar_html, host_bar_html, recurring_issues_html, assignment_group_html,
            chat_stats_state,
            file_input, text_input,
            summary_stats_state,
            # Overview-tab duplicates of the Categorization tab's
            # category bar-list / priority donut - _analyze yields the
            # same two values twice (see its final yield) rather than
            # this chaining a second .then() that reads them back out
            # of category_chart/category_bar_html as inputs.
            overview_category_bar_html, overview_priority_chart,
            source_label_state,
        ]

        _reveal_outputs = [
            hero_html, input_panel, loaded_strip, loaded_strip_html, overview_results,
            tab_categorization, tab_trends, tab_recommendations, tab_qa,
            exec_summary_output,
        ]

        def _wire_post_analysis(event):
            """Everything that runs after _analyze, shared by the Analyze and
            Try-sample-data buttons so both refresh exactly the same views."""
            return event.then(
                # Swap hero/upload for the Loaded strip, show the Overview
                # cards and unlock the result tabs as soon as _analyze is
                # done, so a later refresh step failing can't leave the page
                # looking empty.
                fn=_reveal_after_analysis,
                inputs=[full_results_state, source_label_state],
                outputs=_reveal_outputs,
            ).then(
                fn=_refresh_view,
                inputs=[full_results_state, category_state, min_score_state, page_size_state],
                outputs=[results_table, page_indicator, filtered_results_state, page_state],
            ).then(
                fn=_refresh_categorization_controls,
                inputs=[full_results_state],
                outputs=[category_filter, priority_filter, search_box, review_only, cat_summary_html, detail_drawer],
            ).then(
                fn=_build_chat_index,
                inputs=[full_results_state],
                outputs=[chat_index_state],
            ).then(
                fn=_refresh_trend,
                inputs=[full_results_state, trend_granularity],
                outputs=[trend_chart, resolution_metrics_html],
            ).then(
                fn=_refresh_overview,
                inputs=[full_results_state, summary_stats_state],
                outputs=[summary_md, health_html, attention_html, overview_trend_chart],
            ).then(
                # Recommendations refresh automatically with every new batch.
                # Chained as a separate .then() rather than folded into
                # _analyze's yield so that function's output
                # contract (see _ANALYZE_N_OUTPUTS) is untouched.
                fn=_refresh_recommendations,
                inputs=[full_results_state],
                outputs=[recommendations_html, recommendations_state],
            ).then(
                # Incident Timeline table/KPIs refresh automatically with
                # every new batch, same as Recommendations.
                fn=_refresh_incident_timeline,
                inputs=[full_results_state],
                outputs=[timeline_table, timeline_kpi_html, timeline_aggregate_state],
            ).then(
                fn=_refresh_category_heatmap,
                inputs=[full_results_state],
                outputs=[category_heatmap],
            )

        _wire_post_analysis(analyze_btn.click(
            fn=_analyze,
            inputs=[file_input, text_input],
            outputs=_analyze_outputs,
        ))
        _wire_post_analysis(sample_btn.click(
            fn=_analyze_sample,
            inputs=None,
            outputs=_analyze_outputs,
        ))
        new_analysis_btn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[input_panel],
        )

        timeline_summary_btn.click(
            fn=_generate_timeline_summary,
            inputs=[full_results_state, timeline_aggregate_state],
            outputs=[timeline_summary_output],
        )

        rec_writeup_btn.click(
            fn=_generate_recommendations_writeup,
            inputs=[full_results_state, recommendations_state],
            outputs=[rec_writeup_output],
        )

        exec_summary_btn.click(
            fn=_generate_executive_summary,
            inputs=[full_results_state, summary_stats_state],
            outputs=[exec_summary_output],
        )

        trend_granularity.change(
            fn=_refresh_trend,
            inputs=[full_results_state, trend_granularity],
            outputs=[trend_chart, resolution_metrics_html],
        )

        semantic_recurrence_btn.click(
            fn=_detect_semantic_recurrence,
            inputs=[chat_index_state],
            outputs=[semantic_recurrence_html],
        )

        # Categorization filter bar -> same table/pagination outputs as _refresh_view.
        _filter_inputs = [
            full_results_state, search_box, category_filter, priority_filter,
            review_only, min_score_state, page_size_state,
        ]
        _filter_outputs = [results_table, page_indicator, filtered_results_state, page_state]
        for _ctl in (search_box, category_filter, priority_filter, review_only):
            _ctl.input(fn=_filter_view, inputs=_filter_inputs, outputs=_filter_outputs)

        # Row click -> detail drawer.
        results_table.select(
            fn=_show_row_detail,
            inputs=[filtered_results_state, page_state, page_size_state],
            outputs=[detail_drawer, row_detail_html],
        )
        detail_close_btn.click(fn=_close_row_detail, inputs=None, outputs=[detail_drawer])

        prev_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, -1),
            inputs=[filtered_results_state, page_state, page_size_state],
            outputs=[results_table, page_indicator, page_state],
        )
        next_btn.click(
            fn=lambda filtered_df, page, page_size: _go_to_page(filtered_df, page, page_size, 1),
            inputs=[filtered_results_state, page_state, page_size_state],
            outputs=[results_table, page_indicator, page_state],
        )
        page_size_dropdown.change(
            fn=_change_page_size,
            inputs=[filtered_results_state, page_size_dropdown],
            outputs=[results_table, page_indicator, page_state, page_size_state],
        )

        kb_upload_btn.click(
            fn=_kb_upload,
            inputs=[kb_file_input],
            outputs=[kb_documents_table, kb_document_dropdown, kb_upload_status],
        ).then(
            # Clear the file picker after a successful/attempted upload so
            # a stale selection can't be mistaken for "not yet uploaded"
            # and accidentally re-submitted.
            fn=lambda: gr.update(value=None),
            outputs=[kb_file_input],
        )
        kb_delete_btn.click(
            fn=_kb_delete,
            inputs=[kb_document_dropdown],
            outputs=[kb_documents_table, kb_document_dropdown, kb_manage_status],
        )
        kb_reindex_btn.click(
            fn=_kb_reindex,
            inputs=[kb_document_dropdown],
            outputs=[kb_documents_table, kb_document_dropdown, kb_manage_status],
        )
        demo.load(fn=_kb_refresh, outputs=[kb_documents_table, kb_document_dropdown])

        chat_send.click(
            fn=_chat_respond,
            inputs=[chat_input, chat_history_state, full_results_state, chat_stats_state],
            outputs=[chatbot, chat_history_state, chat_input],
        )
        chat_input.submit(
            fn=_chat_respond,
            inputs=[chat_input, chat_history_state, full_results_state, chat_stats_state],
            outputs=[chatbot, chat_history_state, chat_input],
        )
        chat_clear_btn.click(fn=_chat_clear, outputs=[chatbot, chat_history_state])

    return demo
