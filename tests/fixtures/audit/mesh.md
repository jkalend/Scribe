---
id: AUD-MESH
title: Mesh replay audit
type: audit
status: finding
tags: [mesh, storage]
updated: "2026-08-02"
depends_on: [SYS-QUEUE]
---
# Mesh replay audit

The mesh replay audit found duplicate envelopes during reconnect.
Consumers must deduplicate delivery using sequence numbers.
This audit reports a finding, not a replacement protocol specification.
