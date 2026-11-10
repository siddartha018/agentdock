"""Server entry point: every agent in agents/ becomes a wrapped graph here.

``langgraph.json`` points each graph at ``entry.py:<agent name>``.
"""

from agentdock.platform import wrap
from agentdock.registry import discover, import_agent_module

AGENTS = {}
for _m in discover():
    AGENTS[_m.name] = globals()[_m.name] = wrap(_m, import_agent_module(_m).build())
