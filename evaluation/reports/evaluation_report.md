ITSM AGENT EVALUATION
============================================================
Run: 2026-10-09 03:41   LLM mode: rule_based
NOTE: no LLM configured -> this evaluates the rule_based keyword fallback and keyword
      retrieval. Configure LLM_PROVIDER in .env and re-run to evaluate the real agent.

Preflight: categorization 21/21 tickets categorized as designed
           KB escalation_procedure_p1.md: Indexed (3 chunks)
           KB runbook_database_connection_failures.md: Indexed (4 chunks)
           KB sop_backup_failure_response.md: Indexed (3 chunks)
           KB vpn_troubleshooting_guide.md: Indexed (3 chunks)

AGENT + RAG
------------------------------------------------------------
Total Tests: 28   PASS: 17   FAIL: 11   ERROR: 0
Tool Selection Accuracy:       81.5%  (27 applicable tests)
RAG Retrieval Accuracy:        85.7%  (7 applicable tests)
Answer Accuracy:               60.7%  (28 applicable tests)
Answer Relevance:              75.0%  (28 applicable tests)
Grounded Answers:              71.4%  (28 applicable tests)
Hallucination Rate:            14.3%  (28 applicable tests)
Unnecessary Tool Call Rate:    7.1%  (28 applicable tests)

ID   | Question                                       | Expected                           | Actual                             | Result
--------------------------------------------------------------------------------------------------------------------------------------------
T01  | Give me an overall summary of the incidents.   | get_incident_summary               | get_incident_summary               | PASS
T02  | What is the average worklog score?             | get_incident_summary               | get_incident_summary               | PASS
T03  | What is the average resolution time?           | get_incident_summary               | get_incident_summary               | PASS
T04  | How many P1 incidents are there?               | get_priority_analysis              | get_priority_analysis              | PASS
T05  | What is the priority distribution of the inci… | get_priority_analysis              | get_priority_analysis              | PASS
T06  | What is the most common incident category?     | get_category_analysis              | get_category_analysis              | PASS
T07  | How many Backup related incidents are there?   | get_category_analysis              | search_incidents                   | PASS
T08  | List the top 3 incident categories.            | get_category_analysis              | get_category_analysis              | PASS
T09  | Which server has the most incidents?           | get_server_analysis                | get_server_analysis                | PASS
T10  | Show incidents on server VPN-GW01.             | search_incidents                   | get_server_analysis                | PASS
T11  | What issues keep recurring?                    | get_recurring_issues               | get_recurring_issues               | PASS
T12  | Which servers have recurring database problem… | get_recurring_issues               | get_recurring_issues               | PASS
T13  | What should we do to improve?                  | get_recommendations                | get_recommendations                | PASS
T14  | What needs attention first?                    | get_recommendations                | get_recommendations                | PASS
T15  | What is the documented procedure for database… | search_knowledge_base              | search_knowledge_base              | PASS
T16  | What does the backup SOP say to do after a ba… | search_knowledge_base              | search_knowledge_base              | PASS
T17  | What is the escalation procedure for P1 incid… | search_knowledge_base              | search_knowledge_base              | PASS
T18  | How do I troubleshoot VPN connection problems… | search_knowledge_base              | get_incident_summary               | FAIL
T19  | We have recurring database incidents. What do… | get_recurring_issues+search_knowl… | search_knowledge_base              | FAIL
T20  | How many P1 incidents do we have, and what do… | get_priority_analysis+search_know… | search_knowledge_base              | FAIL
T21  | Backup failures are recurring on our servers.… | get_recurring_issues+search_knowl… | search_knowledge_base              | FAIL
T22  | What is the documented procedure for mainfram… | search_knowledge_base              | search_knowledge_base              | FAIL
T23  | What does our SOP say about SAP HANA system r… | search_knowledge_base              | search_knowledge_base              | FAIL
T24  | What does the runbook say about replacing a f… | search_knowledge_base              | search_knowledge_base              | FAIL
T25  | Make up a runbook for Oracle RAC failover and… | search_knowledge_base              | search_knowledge_base              | FAIL
T26  | What was the CPU temperature of DB-PRD-01 dur… | (no specific tool)                 | get_incident_summary               | FAIL
T27  | How many incidents are there on server XYZ-99… | search_incidents                   | get_server_analysis                | FAIL
T28  | What is the on-call phone number for the DBA … | search_knowledge_base              | get_incident_summary               | FAIL

Why tests did not pass:
  T18 [FAIL]
      - tools: expected ['search_knowledge_base'], got ['get_incident_summary']
      - rag: 'vpn_troubleshooting_guide.md' not retrieved (got nothing)
      - answer: missing 'IKE'
      - relevance: answer does not mention the topic
      - grounding: source document 'vpn_troubleshooting_guide.md' not named in the answer
  T19 [FAIL]
      - tools: expected ['get_recurring_issues', 'search_knowledge_base'], got ['search_knowledge_base']
      - answer: missing 'DB-PRD-01'
  T20 [FAIL]
      - tools: expected ['get_priority_analysis', 'search_knowledge_base'], got ['search_knowledge_base']
      - answer: missing '15 minutes'
  T21 [FAIL]
      - tools: expected ['get_recurring_issues', 'search_knowledge_base'], got ['search_knowledge_base']
      - answer: missing 'BKP-SRV01'
  T22 [FAIL]
      - answer: did not say the information is unavailable / not in the knowledge base
      - relevance: answer does not mention the topic
      - hallucination: presented unrelated KB text as if it answered the question
  T23 [FAIL]
      - answer: did not say the information is unavailable / not in the knowledge base
      - relevance: answer does not mention the topic
      - hallucination: presented unrelated KB text as if it answered the question
  T24 [FAIL]
      - answer: did not say the information is unavailable / not in the knowledge base
      - relevance: answer does not mention the topic
      - hallucination: presented unrelated KB text as if it answered the question
  T25 [FAIL]
      - answer: did not say the information is unavailable / not in the knowledge base
      - hallucination: presented unrelated KB text as if it answered the question
  T26 [FAIL]
      - answer: did not say the information is unavailable / not in the knowledge base
      - relevance: answer does not mention the topic
  T27 [FAIL]
      - answer: did not say the information is unavailable / not in the knowledge base
      - relevance: answer does not mention the topic
  T28 [FAIL]
      - tools: expected ['search_knowledge_base'], got ['get_incident_summary']
      - answer: did not say the information is unavailable / not in the knowledge base
      - relevance: answer does not mention the topic

Unnecessary tool calls:
  T18: also called ['get_incident_summary']
  T28: also called ['get_incident_summary']