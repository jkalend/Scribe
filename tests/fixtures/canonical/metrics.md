---
id: OPS-METRICS
title: Queue observability counters
type: spec
status: accepted
tags: [mesh, observability]
updated: "2026-04-01"
depends_on: [SYS-QUEUE]
---
# Queue observability counters

Queue metrics expose backlog length and oldest envelope age.
Counters contain aggregate totals, never credential values.
An alert fires when the oldest envelope age exceeds ten minutes.
