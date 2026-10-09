# Payments Cluster Failover Runbook (Team A only)

## Procedure
1. Freeze the payments gateway using the command pay-freeze-alpha.
2. Promote the standby node PAYSTBY-A2 to primary.
3. Release the gateway freeze and confirm transaction flow.
