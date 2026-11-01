---
id: SYS-QUEUE
title: Durable queue delivery
type: spec
status: accepted
tags: [mesh, storage]
updated: "2026-01-11"
---
# Durable queue delivery

The queue persists envelopes before acknowledging delivery.
Replay uses monotonically increasing sequence numbers.
Delivery is at least once; consumers deduplicate envelopes.
