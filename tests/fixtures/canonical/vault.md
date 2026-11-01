---
id: SYS-VAULT
title: Local credential vault
type: spec
status: accepted
tags: [security, vault]
updated: "2026-02-01"
---
# Local credential vault

The vault encrypts credentials with a locally derived master key.
Unlock requires the passphrase; there is no remote recovery service.
An offline recovery kit is described in [recovery](recovery.md).
