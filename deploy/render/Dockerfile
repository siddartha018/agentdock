FROM langchain/langgraph-api:3.13

# Postgres + Redis run inside the Space container
RUN apt-get update && apt-get install -y --no-install-recommends postgresql redis-server && rm -rf /var/lib/apt/lists/*



# -- Adding local package . --
ADD . /deps/agentdock
# -- End of local package . --

# -- Installing all local dependencies --
RUN for dep in /deps/*; do             echo "Installing $dep";             if [ -d "$dep" ]; then                 echo "Installing $dep";                 (cd "$dep" && PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir -c /api/constraints.txt -e .);             fi;         done
# -- End of local dependencies install --
ENV LANGGRAPH_STORE='{"ttl": {"default_ttl": 43200, "sweep_interval_minutes": 10}}'
ENV LANGGRAPH_AUTH='{"path": "/deps/agentdock/src/agentdock/auth.py:auth"}'
ENV LANGGRAPH_HTTP='{"app": "/deps/agentdock/src/agentdock/http_app.py:app"}'
ENV LANGGRAPH_CHECKPOINTER='{"ttl": {"strategy": "delete", "default_ttl": 43200, "sweep_interval_minutes": 10}}'
ENV LANGSERVE_GRAPHS='{"concierge": {"path": "/deps/agentdock/src/agentdock/entry.py:concierge", "description": "Front door for employee questions. Works out which specialist agent should answer (HR policy and leave balances, or expense checks) and asks it over A2A on behalf of the employee."}, "expense_check": {"path": "/deps/agentdock/src/agentdock/entry.py:expense_check", "description": "Checks one expense against the travel and expense policy and returns a decision: approve, approve a capped amount, needs a receipt, needs manager approval, or reject. Deterministic rules, no LLM."}, "policy_qa": {"path": "/deps/agentdock/src/agentdock/entry.py:policy_qa", "description": "Answers HR policy questions (vacation, sick leave, parental leave, remote work, expenses, holidays) and looks up leave balances. Employees see only their own balance, managers their direct reports, HR everyone."}}'



# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 uv pip install --system --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --
RUN pip uninstall -y pip setuptools wheel
RUN rm -rf /usr/local/lib/python*/site-packages/pip* /usr/local/lib/python*/site-packages/setuptools* /usr/local/lib/python*/site-packages/wheel* && find /usr/local/bin -name "pip*" -delete || true
RUN rm -rf /usr/lib/python*/site-packages/pip* /usr/lib/python*/site-packages/setuptools* /usr/lib/python*/site-packages/wheel* && find /usr/bin -name "pip*" -delete || true
RUN uv pip uninstall --system pip setuptools wheel && rm /usr/bin/uv /usr/bin/uvx

WORKDIR /deps/agentdock

# Hugging Face runs Spaces as uid 1000
RUN (id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 user) && chown -R 1000:1000 /deps
ENV PORT=7860 \
    DATABASE_URI=postgres://agentdock@127.0.0.1:5432/agentdock?sslmode=disable \
    REDIS_URI=redis://127.0.0.1:6379 \
    N_JOBS_PER_WORKER=8 \
    AGENTDOCK_SERVICE_KEY=dk_dev_svc-concierge \
    AGENTDOCK_RETENTION_KEY=dk_dev_svc-retention \
    AGENTDOCK_AUDIT_LOG=/tmp/agentdock-audit.jsonl \
    AGENTDOCK_FALLBACK=offline
USER 1000
EXPOSE 7860
ENTRYPOINT ["/deps/agentdock/deploy/hf-space/start.sh"]
