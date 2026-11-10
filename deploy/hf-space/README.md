---
title: agentdock
emoji: 🧩
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: LangGraph Agent Server with auth, PII redaction, MCP + A2A
---

# agentdock (production runtime)

The LangGraph Agent Server's production runtime with Postgres and Redis, serving the
agentdock agents over REST, MCP (`/mcp`) and A2A (`/a2a/<assistant_id>`).
Source, docs and the measured results: https://github.com/siddartha018/agentdock

All people and HR data are fake. Dev API keys (`dk_dev_e100`, `dk_dev_m200`, `dk_dev_h300`, ...)
are documented in the repository; they unlock fake identities only.

Postgres and Redis run inside this container, so conversations reset when the Space restarts.
