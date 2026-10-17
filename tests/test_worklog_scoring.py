"""Work-log scoring uses Work Notes + External Info + Internal Info.

Covers: both sources populated, only one populated, neither, conflicting sources,
duplicate/malformed input, the optional LLM pass (labelled sections), and
compatibility with the pre-existing output (scores for worklog-only tickets, schemas,
the CSV-export table). No real LLM is called - the chat model is a local stub.
"""
import asyncio
import json
import logging

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.models.schemas import AnalyzedTicket, TicketRecord, WorklogScore
from app.services import graph_pipeline, pipeline
from app.services.data_ingestion import load_csv
from app.services.persistence import compute_ticket_content_hash
from app.services.scorer import (
    combine_evidence,
    detect_source_conflict,
    heuristic_score,
    prepare_sources,
)
from app.services.validator import normalize_and_validate

ROOT_CAUSE_RESOLUTION = (
    "Root cause: expired TLS certificate. Replaced the certificate and restarted nginx at 10:15 on 2026-08-01; "
    "health check validated OK."
)
EXTERNAL_UPDATE = "Customer was informed at 10:20 that the issue is resolved and the portal is working normally again."
INTERNAL_NOTES = "Investigated nginx logs, found that the certificate had expired. Replaced it and restarted the service."


def _conflict_flags(result):
    return [f for f in result.flags if f.startswith("Possible conflict")]


# --------------------------------------------------------------------------- 1. both populated
def test_both_sources_populated_are_used_together():
    worklog_only = heuristic_score("checked")
    both = heuristic_score("checked", EXTERNAL_UPDATE, INTERNAL_NOTES)
    assert both.score > worklog_only.score
    assert both.breakdown["root_cause"] == 25 and both.breakdown["resolution"] == 25
    assert "Root cause not clearly documented." not in both.flags
    assert 0 <= both.score <= 100


def test_evidence_can_be_spread_across_the_two_sources():
    # root cause only in Internal Info, resolution wording only in External Info
    r = heuristic_score("", "The fix was applied and the service restored for users.", "Root cause: disk full on /var.")
    assert r.breakdown["root_cause"] == 25 and r.breakdown["resolution"] == 25


# --------------------------------------------------------------------------- 2/3. one source only
@pytest.mark.parametrize("which", ["worklog", "external", "internal"])
def test_a_single_populated_source_scores_the_same_wherever_it_sits(which):
    args = {"worklog": ("", "", ""), "external": ("", "", ""), "internal": ("", "", "")}
    text = ROOT_CAUSE_RESOLUTION
    args["worklog"] = (text, "", "")
    args["external"] = ("", text, "")
    args["internal"] = ("", "", text)
    reference = heuristic_score(text)
    r = heuristic_score(*args[which])
    assert (r.score, r.breakdown) == (reference.score, reference.breakdown)
    assert "No worklog provided." not in r.flags


def test_missing_source_is_not_penalised_or_flagged():
    r = heuristic_score(ROOT_CAUSE_RESOLUTION, "", "")
    assert r.score == 100
    assert not any("External" in f or "Internal" in f or "missing" in f.lower() for f in r.flags)


# --------------------------------------------------------------------------- 4. nothing at all
@pytest.mark.parametrize("empty", [("", "", ""), (None, None, None), (float("nan"),) * 3, ("  ", "\n\n", "\t")])
def test_all_sources_empty_scores_zero(empty):
    r = heuristic_score(*empty)
    assert r.score == 0 and r.flags == ["No worklog provided."]
    assert r.breakdown == {"completeness": 0, "root_cause": 0, "resolution": 0, "professionalism": 0}


def test_placeholder_only_input_is_still_flagged_like_before():
    for args in [("na", "", ""), ("", "TEST", ""), ("", "", "tbd")]:
        r = heuristic_score(*args)
        assert "Placeholder/empty worklog content." in r.flags
        assert r.breakdown["completeness"] == 0


def test_placeholder_in_one_source_does_not_hide_real_evidence_in_another():
    r = heuristic_score("na", ROOT_CAUSE_RESOLUTION, "n/a")
    assert r.score == heuristic_score(ROOT_CAUSE_RESOLUTION).score == 100


# --------------------------------------------------------------------------- 5. conflicts
def test_conflicting_sources_are_flagged_not_resolved():
    r = heuristic_score(
        ROOT_CAUSE_RESOLUTION,
        "Update to customer: issue resolved, service restored.",
        "Issue still persists on node 2, no fix found yet.",
    )
    flags = _conflict_flags(r)
    assert flags and any("External Info" in f and "Internal Info" in f for f in flags)
    # the conflict is reported only - the score is the plain rubric score of the evidence
    no_conflict = heuristic_score(
        ROOT_CAUSE_RESOLUTION, "Update to customer: issue resolved, service restored.", "Checked node 2 and confirmed."
    )
    assert not _conflict_flags(no_conflict)


def test_conflict_detection_is_symmetric_neither_source_is_trusted_more():
    a = heuristic_score("", "The issue is resolved and service restored.", "Issue still persists, not fixed.")
    b = heuristic_score("", "Issue still persists, not fixed.", "The issue is resolved and service restored.")
    assert a.score == b.score
    assert len(_conflict_flags(a)) == len(_conflict_flags(b)) == 1


def test_agreement_between_sources_is_not_a_conflict():
    assert not _conflict_flags(heuristic_score("", "Issue resolved.", "Fixed and restored the service."))
    assert not _conflict_flags(heuristic_score("", "Issue not resolved yet.", "Issue still persists."))


def test_detect_source_conflict_never_raises_on_bad_input():
    assert detect_source_conflict({}) == []
    assert detect_source_conflict({"worklog": None, "external_info": 5}) == []


# --------------------------------------------------------------------------- 6. duplicates / malformed
def test_duplicate_entries_do_not_inflate_the_score():
    once = heuristic_score("Restarted the service.")
    repeated = heuristic_score("Restarted the service.\nrestarted  the service.\nRESTARTED THE SERVICE.\n")
    assert once.score == repeated.score and once.breakdown == repeated.breakdown


def test_same_note_copied_into_both_sources_counts_once():
    note = "Restarted the service."
    assert heuristic_score(note).score == heuristic_score(note, note, note).score


@pytest.mark.parametrize("bad", [None, float("nan"), 12345, ["a", "b"], {"x": 1}, b"bytes note", "\x00\x01 odd \x7f chars", "a\r\nb\rc"])
def test_malformed_values_never_raise(bad):
    for args in [(bad, "", ""), ("", bad, ""), ("", "", bad), (bad, bad, bad)]:
        r = heuristic_score(*args)
        assert isinstance(r, WorklogScore) and 0 <= r.score <= 100


def test_very_long_input_is_handled():
    r = heuristic_score("word " * 50_000, "x" * 100_000, "")
    assert 0 <= r.score <= 100


def test_prepare_sources_keeps_the_sources_separate_and_cleans_them():
    s = prepare_sources("  Checked  logs \r\nChecked logs\n\n", "n/a", None)
    assert s == {"worklog": "Checked logs", "external_info": "", "internal_info": ""}
    assert combine_evidence({"worklog": "A", "external_info": "a", "internal_info": "B"}) == "A\nB"


# --------------------------------------------------------------------------- 7. compatibility
# Scores produced by the previous implementation (single-argument heuristic_score).
LEGACY = {
    "Root cause: log rotation was disabled. Resolved by clearing old logs and re-enabling logrotate. Extended /var by 50GB via LVM.": 100,
    "checked": 40,
    "Investigated source IP, blocked at firewall, reset affected account password, escalated to security team for full audit.": 80,
    "na": 35,
    "TEST": 25,
    "": 0,
    "FIX IT NOW PLEASE URGENT URGENT URGENT URGENT URGENT URGENT": 40,
    "Root cause: bad cert. Replaced cert and restarted nginx at 10:15 on 2026-08-01, validated OK.": 100,
}


@pytest.mark.parametrize("worklog,expected", list(LEGACY.items()))
def test_worklog_only_scores_match_the_previous_implementation(worklog, expected):
    assert heuristic_score(worklog).score == expected
    assert heuristic_score(worklog, "", "").score == expected


def test_scoring_output_schema_is_unchanged():
    assert set(WorklogScore.model_fields) == {"ticket_id", "score", "breakdown", "flags"}
    assert "internal_info" not in AnalyzedTicket.model_fields  # results/export schema untouched
    assert {"worklog_score", "worklog_flags", "external_info"} <= set(AnalyzedTicket.model_fields)


def test_internal_info_column_is_recognised_by_ingestion_and_validation():
    csv = (
        "Incident Number,Short Description,Work Notes,External Info,Internal Info\n"
        'INC9001,VPN down,checked,"Customer told it is resolved","Root cause: expired cert. Replaced it."\n'
        "INC9002,Disk full,Cleared logs,,\n"
    ).encode()
    df = load_csv(csv)
    assert {"worklog", "external_info", "internal_info"} <= set(df.columns)
    records, rejected = normalize_and_validate(df)
    assert not rejected
    assert records[0].internal_info.startswith("Root cause") and records[1].internal_info == ""
    assert isinstance(records[0], TicketRecord)
    for alias in ("Internal Information",):
        assert "internal_info" in load_csv(f"Incident Number,Description,{alias}\nINC1,x,y\n".encode()).columns


def test_files_without_the_new_column_still_load_and_score():
    df = load_csv(b"Incident Number,Short Description,Work Notes\nINC1,Disk full,Cleared old logs and resolved it.\n")
    records, _ = normalize_and_validate(df)
    assert records[0].internal_info == ""
    assert heuristic_score(records[0].worklog, records[0].external_info, records[0].internal_info).score > 0


def test_content_hash_reacts_to_internal_info_and_stays_deterministic():
    base = compute_ticket_content_hash("s", "d", "w", "sub", "ext")
    assert base == compute_ticket_content_hash("s", "d", "w", "sub", "ext", "")
    assert base != compute_ticket_content_hash("s", "d", "w", "sub", "ext", "internal note")
    assert compute_ticket_content_hash("s", "d", "w", "sub", "ext", "x") == compute_ticket_content_hash("s", "d", "w", "sub", "ext", "x")


def test_pipeline_end_to_end_with_external_and_internal_info():
    csv = (
        "Incident Number,Short Description,Description,Work Notes,External Info,Internal Info,Priority,State\n"
        'INC7001,Cert expired on portal,Users see TLS errors,checked,"Customer informed, service restored","Root cause: expired cert. Replaced the cert and restarted nginx.",P2,Closed\n'
        "INC7002,Disk full on app server,/var at 95%,,,,P3,Closed\n"
        'INC7003,VPN down,Tunnel fails,na,"Issue resolved, service restored","Issue still persists, no fix found yet",P2,In Progress\n'
    ).encode()
    result = asyncio.run(pipeline.run_pipeline_from_bytes("t.csv", csv))
    by_id = {t.ticket_id: t for t in result.results}
    assert set(by_id) == {"INC7001", "INC7002", "INC7003"}
    assert by_id["INC7001"].worklog_score > 40            # 'checked' alone scores 40
    assert by_id["INC7002"].worklog_score == 0            # nothing documented anywhere
    assert any("Possible conflict" in f for f in by_id["INC7003"].worklog_flags)
    assert by_id["INC7001"].worklog == "checked"          # displayed worklog column untouched
    assert 0 <= result.average_worklog_score <= 100
    from ui.helpers import _results_to_full_dataframe
    df = _results_to_full_dataframe(result)
    assert {"Worklog Notes", "External Info", "Worklog Score", "Worklog Rating", "Worklog Flags"} <= set(df.columns)
    assert "Internal Info" not in df.columns              # export columns unchanged
    assert df["Worklog Score"].between(0, 100).all()


# --------------------------------------------------------------------------- LLM pass (stubbed model)
class _Stub:
    """Chat-model stand-in: records the rendered prompt, returns canned content."""

    def __init__(self, content=None, error=None):
        self.prompts, self.content, self.error = [], content, error

    def model(self):
        def run(prompt_value):
            self.prompts.append(prompt_value.to_string())
            if self.error:
                raise self.error
            return AIMessage(content=self.content)

        return RunnableLambda(run)


def _state(worklog="", external="", internal=""):
    return {
        "ticket_id": "INC1", "text": "VPN down", "worklog": worklog, "external_info": external,
        "internal_info": internal, "category": "Network issues", "category_confidence": 1.0,
        "category_method": "keyword", "worklog_score": 0, "worklog_flags": [], "error": None,
    }


@pytest.fixture
def llm_on(monkeypatch):
    monkeypatch.setattr(graph_pipeline.settings, "ENABLE_LLM_WORKLOG_SCORING", True)


def test_llm_prompt_has_separate_labelled_sections_and_one_call(llm_on):
    stub = _Stub(json.dumps({"score": 90, "breakdown": {}, "flags": ["Conflict: External vs Internal"]}))
    out = asyncio.run(graph_pipeline.score_node(_state("checked", EXTERNAL_UPDATE, INTERNAL_NOTES), stub.model()))
    assert len(stub.prompts) == 1
    p = stub.prompts[0]
    assert "Work Notes: checked" in p
    assert "External Info: " + EXTERNAL_UPDATE in p and "Internal Info: " + INTERNAL_NOTES in p
    assert p.index("External Info:") < p.index("Internal Info:")
    assert "Conflict: External vs Internal" in out["worklog_flags"]
    assert 0 <= out["worklog_score"] <= 100


def test_llm_prompt_marks_empty_sections_instead_of_dropping_them(llm_on):
    stub = _Stub(json.dumps({"score": 70, "flags": []}))
    asyncio.run(graph_pipeline.score_node(_state("", "", INTERNAL_NOTES), stub.model()))
    p = stub.prompts[0]
    assert "Work Notes: (none provided)" in p and "External Info: (none provided)" in p
    assert "Internal Info: " + INTERNAL_NOTES in p


def test_llm_prompt_tells_model_not_to_penalise_missing_sources_or_invent_details(llm_on):
    stub = _Stub(json.dumps({"score": 70, "flags": []}))
    asyncio.run(graph_pipeline.score_node(_state("", EXTERNAL_UPDATE, ""), stub.model()))
    p = stub.prompts[0].lower()
    assert "not a defect" in p and "never invent" in p and "contradict" in p and "less reliable" in p


def test_llm_is_not_called_when_there_is_no_usable_evidence_or_when_disabled(monkeypatch):
    stub = _Stub(json.dumps({"score": 70, "flags": []}))
    monkeypatch.setattr(graph_pipeline.settings, "ENABLE_LLM_WORKLOG_SCORING", True)
    out = asyncio.run(graph_pipeline.score_node(_state("na", "", "TEST"), stub.model()))
    assert stub.prompts == [] and out["worklog_score"] == heuristic_score("na", "", "TEST").score
    monkeypatch.setattr(graph_pipeline.settings, "ENABLE_LLM_WORKLOG_SCORING", False)
    asyncio.run(graph_pipeline.score_node(_state("", "", INTERNAL_NOTES), stub.model()))
    assert stub.prompts == []


@pytest.mark.parametrize(
    "content",
    ["not json at all", json.dumps({"score": "high"}), json.dumps({"score": True}), json.dumps([1, 2])],
)
def test_bad_llm_output_falls_back_to_the_heuristic_score(llm_on, content):
    stub = _Stub(content)
    out = asyncio.run(graph_pipeline.score_node(_state("", EXTERNAL_UPDATE, INTERNAL_NOTES), stub.model()))
    assert out["worklog_score"] == heuristic_score("", EXTERNAL_UPDATE, INTERNAL_NOTES).score


def test_llm_flags_of_the_wrong_type_are_ignored(llm_on):
    stub = _Stub(json.dumps({"score": 80, "flags": "a string, not a list"}))
    out = asyncio.run(graph_pipeline.score_node(_state("", EXTERNAL_UPDATE, INTERNAL_NOTES), stub.model()))
    assert all(len(f) > 1 for f in out["worklog_flags"])


def test_llm_failure_degrades_gracefully_and_does_not_log_note_text(llm_on, caplog):
    secret = "SECRET-TOKEN-abc123 internal note"
    stub = _Stub(error=RuntimeError("model unavailable"))
    with caplog.at_level(logging.DEBUG):
        out = asyncio.run(graph_pipeline.score_node(_state("", "", secret + " root cause: x. fixed."), stub.model()))
    assert out["worklog_score"] == heuristic_score("", "", secret + " root cause: x. fixed.").score
    assert "SECRET-TOKEN-abc123" not in caplog.text


def test_score_node_still_works_for_states_without_the_new_keys(llm_on):
    legacy = _state("Root cause: bad cert. Replaced cert and restarted nginx at 10:15 on 2026-08-01, validated OK.")
    legacy.pop("external_info"), legacy.pop("internal_info")
    out = asyncio.run(graph_pipeline.score_node(legacy, None))
    assert out["worklog_score"] == 100


def test_llm_blend_is_unchanged_average_of_heuristic_and_model(llm_on):
    stub = _Stub(json.dumps({"score": 60, "flags": []}))
    h = heuristic_score("", EXTERNAL_UPDATE, INTERNAL_NOTES).score
    out = asyncio.run(graph_pipeline.score_node(_state("", EXTERNAL_UPDATE, INTERNAL_NOTES), stub.model()))
    assert out["worklog_score"] == max(0, min(100, round((h + 60) / 2)))


def test_prompt_template_variables_are_all_supplied():
    assert set(graph_pipeline._SCORE_PROMPT.input_variables) == {"context", "worklog", "external_info", "internal_info"}
