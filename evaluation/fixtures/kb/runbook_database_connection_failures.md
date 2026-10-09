# Database Connection Failure Runbook

## Scope
Applies to production databases when applications report connection timeouts or "cannot connect" errors.

## Immediate Checks
1. Confirm the database listener is running and reachable on port 1521 from the application server.
2. Check the connection pool service dbpool-svc on the application tier for exhausted connections.
3. Review the last 30 minutes of the database alert log for ORA- errors.

## Remediation Steps
1. Restart the connection pool service dbpool-svc during a low-traffic window.
2. If the listener is down, restart it using the standard DBA procedure and verify with a test connection.
3. If the issue repeats within 24 hours, raise a Problem record for root cause analysis.

## Escalation
Escalate to the DBA on-call engineer if the connection failures last longer than 30 minutes.
