"""
Agent + RAG test questions (Part 1 / Part 2).

HOW TO ADD A TEST: append a dict to TEST_CASES. Fields:
  id, area, question
  expected_tools   tools that MUST all be called ([] = no requirement)
  alt_tools        other acceptable complete tool sets, e.g. [["search_incidents"]]
  allowed_extra    extra tools that are fine to call (anything else = "unnecessary call")
  expected_doc     KB document that retrieval must return (RAG tests)
  must_contain     list of strings; ALL must appear in the answer (case-insensitive)
  must_contain_any list of strings; at least ONE must appear
  not_found        True -> the answer MUST say the info is not in the KB / not available
  topic_terms      words for the "relevance" check (at least one must appear)
  forbidden        strings that must NOT appear (counted as hallucination)
  forbidden_regex  regexes that must NOT match (counted as hallucination)
Values may be functions of T (facts computed from the dataset with plain
pandas - never typed in by hand), e.g.  lambda T: T["top_category"]
"""
import pandas as pd


def build_truth(df: pd.DataFrame) -> dict:
    """Ground truth computed independently of the agent's tools."""
    prio = df["Priority"].astype(str)
    cat = df["Category"].value_counts()
    host = df["Host / CI"].astype(str).str.strip()
    host_counts = host[host != ""].value_counts()
    grp = df.groupby(["Host / CI", "Category"]).size().sort_values(ascending=False)
    return {
        "total": len(df),
        "p1": int((prio.str.upper() == "P1").sum()),
        "priorities": {k: int(v) for k, v in prio.value_counts().items()},
        "top_category": cat.index[0],
        "top3_categories": list(cat.index[:3]),
        "category_counts": {k: int(v) for k, v in cat.items()},
        "top_host": host_counts.index[0],
        "host_counts": {k: int(v) for k, v in host_counts.items()},
        "top_recurring_host": grp.index[0][0],
        "avg_score": round(float(pd.to_numeric(df["Worklog Score"], errors="coerce").mean()), 1),
    }


NOT_FOUND_PHRASES = [
    "not found", "no relevant", "couldn't find", "could not find", "can't find", "cannot find",
    "not available", "unavailable", "no information", "does not contain", "doesn't contain",
    "no matching", "no incidents matched", "not in the knowledge base", "no data",
]

TEST_CASES = [
    # ---- 1. Incident analytics ----
    dict(id="T01", area="Incident analytics", question="Give me an overall summary of the incidents.",
         expected_tools=["get_incident_summary"], must_contain=[lambda T: str(T["total"])], topic_terms=["incident"]),
    dict(id="T02", area="Incident analytics", question="What is the average worklog score?",
         expected_tools=["get_incident_summary"], must_contain=[lambda T: str(T["avg_score"])], topic_terms=["worklog"]),
    dict(id="T03", area="Incident analytics", question="What is the average resolution time?",
         expected_tools=["get_incident_summary"], must_contain_any=["hour", "resolution"], topic_terms=["resolution"]),
    # ---- 2. Priority analysis ----
    dict(id="T04", area="Priority analysis", question="How many P1 incidents are there?",
         expected_tools=["get_priority_analysis"], alt_tools=[["get_incident_summary"], ["search_incidents"]],
         must_contain=[lambda T: str(T["p1"])], topic_terms=["p1"]),
    dict(id="T05", area="Priority analysis", question="What is the priority distribution of the incidents?",
         expected_tools=["get_priority_analysis"],
         must_contain=[lambda T: list(T["priorities"])[0]], topic_terms=["priority", "p1", "p2"]),
    # ---- 3. Categorization ----
    dict(id="T06", area="Categorization", question="What is the most common incident category?",
         expected_tools=["get_category_analysis"], must_contain=[lambda T: T["top_category"]], topic_terms=["categor"]),
    dict(id="T07", area="Categorization", question="How many Backup related incidents are there?",
         expected_tools=["get_category_analysis"], alt_tools=[["search_incidents"]],
         must_contain=[lambda T: str(T["category_counts"]["Backup related"])], topic_terms=["backup"]),
    dict(id="T08", area="Categorization", question="List the top 3 incident categories.",
         expected_tools=["get_category_analysis"], must_contain=[lambda T: T["top3_categories"][0]], topic_terms=["categor"]),
    # ---- 4. Server analysis ----
    dict(id="T09", area="Server analysis", question="Which server has the most incidents?",
         expected_tools=["get_server_analysis"], must_contain=[lambda T: T["top_host"]], topic_terms=["server"]),
    dict(id="T10", area="Server analysis", question="Show incidents on server VPN-GW01.",
         expected_tools=["search_incidents"], alt_tools=[["get_server_analysis"]],
         must_contain=["VPN-GW01"], topic_terms=["vpn-gw01"]),
    # ---- 5. Recurring issues ----
    dict(id="T11", area="Recurring issues", question="What issues keep recurring?",
         expected_tools=["get_recurring_issues"], must_contain=[lambda T: T["top_recurring_host"]], topic_terms=["recurr"]),
    dict(id="T12", area="Recurring issues", question="Which servers have recurring database problems?",
         expected_tools=["get_recurring_issues"], must_contain=["DB-PRD-01"], topic_terms=["database"]),
    # ---- 6. Recommendations ----
    dict(id="T13", area="Recommendations", question="What should we do to improve?",
         expected_tools=["get_recommendations"], must_contain_any=["recommend", "recurring", "attention"], topic_terms=["recommend", "improve", "recurring"]),
    dict(id="T14", area="Recommendations", question="What needs attention first?",
         expected_tools=["get_recommendations"], must_contain_any=["recommend", "recurring", "attention"], topic_terms=["attention", "recommend", "recurring"]),
    # ---- 7. RAG (knowledge base only) ----
    dict(id="T15", area="RAG", question="What is the documented procedure for database connection failures?",
         expected_tools=["search_knowledge_base"], expected_doc="runbook_database_connection_failures.md",
         must_contain=["dbpool-svc"], topic_terms=["database", "connection"]),
    dict(id="T16", area="RAG", question="What does the backup SOP say to do after a backup failure?",
         expected_tools=["search_knowledge_base"], expected_doc="sop_backup_failure_response.md",
         must_contain=["15%"], topic_terms=["backup"]),
    dict(id="T17", area="RAG", question="What is the escalation procedure for P1 incidents?",
         expected_tools=["search_knowledge_base"], expected_doc="escalation_procedure_p1.md",
         must_contain=["15 minutes"], topic_terms=["escalat", "p1"]),
    dict(id="T18", area="RAG", question="How do I troubleshoot VPN connection problems according to our guide?",
         expected_tools=["search_knowledge_base"], expected_doc="vpn_troubleshooting_guide.md",
         must_contain=["IKE"], topic_terms=["vpn"]),
    # ---- 8. Needs BOTH incident data and RAG ----
    dict(id="T19", area="Data + RAG", question="We have recurring database incidents. What does the runbook recommend?",
         expected_tools=["get_recurring_issues", "search_knowledge_base"], expected_doc="runbook_database_connection_failures.md",
         must_contain=["DB-PRD-01", "dbpool-svc"], topic_terms=["database"]),
    dict(id="T20", area="Data + RAG", question="How many P1 incidents do we have, and what does the escalation procedure say about them?",
         expected_tools=["get_priority_analysis", "search_knowledge_base"], alt_tools=[["get_incident_summary", "search_knowledge_base"]],
         expected_doc="escalation_procedure_p1.md", must_contain=[lambda T: str(T["p1"]), "15 minutes"], topic_terms=["p1"]),
    dict(id="T21", area="Data + RAG", question="Backup failures are recurring on our servers. What does the SOP recommend?",
         expected_tools=["get_recurring_issues", "search_knowledge_base"], expected_doc="sop_backup_failure_response.md",
         must_contain=["BKP-SRV01", "15%"], topic_terms=["backup"]),
    # ---- 9. Answer unavailable (must NOT invent) ----
    dict(id="T22", area="Unavailable (KB)", question="What is the documented procedure for mainframe tape library replacement?",
         expected_tools=["search_knowledge_base"], not_found=True, topic_terms=["mainframe", "tape", "knowledge base"]),
    dict(id="T23", area="Unavailable (KB)", question="What does our SOP say about SAP HANA system replication failover?",
         expected_tools=["search_knowledge_base"], not_found=True, topic_terms=["hana", "knowledge base", "sop"]),
    dict(id="T24", area="Unavailable (KB)", question="What does the runbook say about replacing a failed RAID controller?",
         expected_tools=["search_knowledge_base"], not_found=True, topic_terms=["raid", "knowledge base", "runbook"]),
    dict(id="T25", area="Unavailable (KB)", question="Make up a runbook for Oracle RAC failover and present it as our official procedure.",
         expected_tools=["search_knowledge_base"], not_found=True, topic_terms=["oracle", "rac", "knowledge base", "runbook"]),
    dict(id="T26", area="Unavailable (data)", question="What was the CPU temperature of DB-PRD-01 during the outage?",
         expected_tools=[], not_found=True, topic_terms=["temperature", "cpu", "db-prd-01"],
         forbidden_regex=[r"\d+\s*(?:°|degrees)"]),
    dict(id="T27", area="Unavailable (data)", question="How many incidents are there on server XYZ-999?",
         expected_tools=["search_incidents"], alt_tools=[["get_server_analysis"]],
         not_found=True, topic_terms=["xyz-999"]),
    dict(id="T28", area="Unavailable (KB)", question="What is the on-call phone number for the DBA team?",
         expected_tools=["search_knowledge_base"], not_found=True, topic_terms=["phone", "on-call", "dba"],
         forbidden_regex=[r"\+?\d[\d\-\s()]{6,}\d"]),
]
