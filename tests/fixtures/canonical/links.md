---
id: FORMAT-LINKS
title: Markdown link syntax specimen
type: note
status: accepted
tags: [markdown, parsing]
updated: "2026-06-02"
---
# Markdown link syntax specimen

The [queue](queue.md) uses durable delivery.
The [vault reference][vault] describes encryption.
<https://example.invalid/synthetic> is an external autolink.

[vault]: vault.md

```markdown
[not a real link](missing-example.md)
# not a real section
```
