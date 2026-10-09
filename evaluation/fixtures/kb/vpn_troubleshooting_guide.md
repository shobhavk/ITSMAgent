# VPN Troubleshooting Guide

## Symptoms
Remote users cannot establish a VPN tunnel to gateway VPN-GW01.

## Troubleshooting Steps
1. Verify the user credentials and MFA registration.
2. Check IKE phase 1 negotiation in the gateway logs for failures.
3. Check the gateway certificate expiry date and renew if it has expired.
4. Confirm the user's ISP is not blocking UDP port 500 and 4500.

## Escalation
Escalate to the Network Team if more than 10 users are affected.
