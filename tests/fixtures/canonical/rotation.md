---
id: SEC-ROTATE
title: Credential key rotation
type: runbook
status: accepted
tags: [security, rotation]
updated: "2026-03-04"
depends_on: [SYS-VAULT]
---
# Credential key rotation

Rotate the wrapping key after a suspected workstation compromise.
Rewrap existing credential records without changing user passwords.
Record completion in the local audit journal.
