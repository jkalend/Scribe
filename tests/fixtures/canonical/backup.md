---
id: OPS-BACKUP
title: Snapshot backup procedure
type: runbook
status: accepted
tags: [storage, backup]
updated: "2026-03-01"
depends_on: [SYS-VAULT]
---
# Snapshot backup procedure

Snapshot backups run nightly and retain seven daily copies.
A monthly restore drill checks decryptability on a spare machine.
Backups are local encrypted archives, not a cloud synchronization service.
