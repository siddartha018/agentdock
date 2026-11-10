"""``dock``: the one command teams use.

    dock new <name> --kind react|workflow --team <team>   scaffold an agent
    dock ls                                              list agents and their endpoints
    dock check                                           validate manifests, graphs and eval files
    dock eval [--agent X] [--update-baseline]            run evals, enforce the gate
    dock serve [--port 2024]                             gate, then start the Agent Server
    dock audit [--tail N] [--verify]                     read the audit log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

from agentdock import audit
from agentdock.identity import ROOT
from agentdock.registry import AGENTS_DIR, KINDS, NAME_RE, ManifestError, discover, write_langgraph_config


def _settings() -> dict:
    from agentdock.evals import platform_settings

    return platform_settings()


def cmd_new(args: argparse.Namespace) -> int:
    if not NAME_RE.match(args.name):
        print(f"name must be snake_case (3-41 chars): {args.name!r}")
        return 2
    dest = AGENTS_DIR / args.name
    if dest.exists():
        print(f"{dest} already exists")
        return 2
    started = time.perf_counter()
    template = resources.files("agentdock") / "templates" / args.kind
    dest.mkdir(parents=True)
    for fname in ("graph.py", "evals.jsonl"):
        (dest / fname).write_text(template.joinpath(fname).read_text().replace("__NAME__", args.name))
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    description = args.description or f"{args.name.replace('_', ' ').capitalize()} agent owned by {args.team}. Replace this description."
    (dest / "agent.toml").write_text(
        f'name = "{args.name}"\n'
        f"description = {json.dumps(description)}\n"
        f'owner_team = "{args.team}"\n'
        f'kind = "{args.kind}"\n'
        f"allowed_roles = {json.dumps(roles)}\n\n"
        "[gate]\nmin_pass_rate = 0.9\nmin_cases = 5\n"
    )
    write_langgraph_config(discover(), _settings()["retention_minutes"])
    print(f"Created agents/{args.name}/ ({args.kind}) in {time.perf_counter() - started:.2f}s")
    print(f"  edit agents/{args.name}/graph.py, then: dock eval --agent {args.name} && dock serve")
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    for m in discover():
        print(f"{m.name:16} {m.kind:9} team={m.owner_team:16} roles={','.join(m.allowed_roles)}")
        print(f"{'':16} MCP tool  {base}/mcp  (tool name: {m.name})")
        print(f"{'':16} A2A card  {base}/a2a/<assistant_id>/.well-known/agent-card.json")
    return 0


def check() -> list[str]:
    """Static checks a team's agent must pass before anything runs."""
    from agentdock.evals import load_cases
    from agentdock.platform import wrap
    from agentdock.registry import import_agent_module

    problems: list[str] = []
    try:
        manifests = discover()
    except ManifestError as exc:
        return [str(exc)]
    if not manifests:
        return ["no agents in agents/"]
    for m in manifests:
        try:
            wrap(m, import_agent_module(m).build())
        except Exception as exc:
            problems.append(f"{m.name}: graph does not build: {type(exc).__name__}: {exc}")
            continue
        try:
            cases = load_cases(m.evals_file)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if len(cases) < m.gate.min_cases:
            problems.append(f"{m.name}: {len(cases)} eval cases, needs at least {m.gate.min_cases}")
        if not any("not_contains" in c.expect or c.expect.get("denied") for c in cases):
            problems.append(f"{m.name}: needs at least one negative case (not_contains or denied)")
    return problems


def cmd_check(args: argparse.Namespace) -> int:
    problems = check()
    if not problems:
        write_langgraph_config(discover(), _settings()["retention_minutes"])
        print(f"check: OK ({len(discover())} agents); langgraph.json regenerated")
        return 0
    print("check: FAILED")
    for p in problems:
        print(f"  - {p}")
    return 1


async def _eval(agent: str | None, update_baseline: bool, quiet: bool = False, pace: float = 0.0) -> int:
    from agentdock.entry import AGENTS
    from agentdock.evals import gate, load_baseline, run_agent, save_baseline, save_results
    from agentdock.models import model_name

    manifests = {m.name: m for m in discover()}
    names = [agent] if agent else list(manifests)
    unknown = [n for n in names if n not in manifests]
    if unknown:
        print(f"unknown agent(s): {unknown}")
        return 2
    reports = [await run_agent(manifests[n], AGENTS[n], pace) for n in names]
    for r in reports:
        print(f"{r.agent:16} {r.passed}/{r.cases} passed ({r.pass_rate:.0%}, floor {r.min_pass_rate:.0%})")
        for c in r.results:
            if not c.passed or not quiet:
                mark = "ok  " if c.passed else "FAIL"
                print(f"   {mark} {c.id:28} {c.latency_ms:7.1f} ms  {'; '.join(c.failures)}")
    path = save_results(reports, model_name())
    if update_baseline:
        save_baseline(reports)
        print(f"baseline updated ({path.name} saved)")
        return 0
    problems = gate(reports, load_baseline(), manifests)
    if problems:
        print("GATE: BLOCKED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("GATE: green")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    return asyncio.run(_eval(args.agent, args.update_baseline, pace=args.pace))


def cmd_serve(args: argparse.Namespace) -> int:
    problems = check()
    if problems:
        print("serve blocked by check:")
        for p in problems:
            print(f"  - {p}")
        return 1
    if not args.skip_gate and asyncio.run(_eval(None, False, quiet=True)) != 0:
        print("serve blocked by the eval gate")
        return 1
    write_langgraph_config(discover(), _settings()["retention_minutes"])
    env_file = ROOT / ".env"
    if not env_file.exists():
        env_file.write_text("# Local settings for `dock serve` (gitignored).\n")
    exe = shutil.which("langgraph", path=str(Path(sys.executable).parent)) or "langgraph"
    # Agents call agents (A2A): a run waiting on a sub-run needs a free slot, or
    # with one slot the server deadlocks. Size this above the deepest call chain.
    cmd = [
        exe, "dev", "--no-browser", "--port", str(args.port),
        "--config", str(ROOT / "langgraph.json"), "--n-jobs-per-worker", str(args.jobs),
    ]
    if args.no_reload:
        cmd.append("--no-reload")
    print("starting:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT, env=os.environ.copy())


def cmd_audit(args: argparse.Namespace) -> int:
    if args.verify:
        ok, bad = audit.verify()
        n = len(audit.read())
        print(f"audit log: {n} records, chain {'intact' if ok else f'BROKEN at record {bad}'}")
        return 0 if ok else 1
    for rec in audit.read()[-args.tail :]:
        flags = []
        if rec.get("denied"):
            flags.append(f"DENIED:{rec['denied']}")
        if rec.get("redacted_in") or rec.get("redacted_out"):
            flags.append(f"redacted in={rec.get('redacted_in') or {}} out={rec.get('redacted_out') or {}}")
        if rec.get("error"):
            flags.append(f"ERROR:{rec['error']}")
        via = f" via {rec['via']}" if rec.get("via") else ""
        print(
            f"{rec['ts']} {rec['agent']:14} {rec.get('caller')}{via} tools={rec.get('tools', [])} "
            f"{rec.get('latency_ms')}ms {' '.join(flags)}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dock", description="Ship LangGraph agents on the paved road.")
    sub = p.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("new", help="scaffold an agent")
    n.add_argument("name")
    n.add_argument("--kind", choices=sorted(KINDS), default="react")
    n.add_argument("--team", required=True)
    n.add_argument("--roles", default="employee,manager,hr")
    n.add_argument("--description")
    n.set_defaults(fn=cmd_new)

    ls = sub.add_parser("ls", help="list agents and endpoints")
    ls.add_argument("--url", default="http://127.0.0.1:2024")
    ls.set_defaults(fn=cmd_ls)

    sub.add_parser("check", help="validate agents").set_defaults(fn=cmd_check)

    e = sub.add_parser("eval", help="run evals and the gate")
    e.add_argument("--agent")
    e.add_argument("--update-baseline", action="store_true")
    e.add_argument("--pace", type=float, default=0.0, help="seconds between cases (free-tier rate limits)")
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("serve", help="gate, then start the Agent Server")
    s.add_argument("--port", type=int, default=2024)
    s.add_argument("--jobs", type=int, default=8, help="concurrent runs (must exceed A2A call depth)")
    s.add_argument("--skip-gate", action="store_true", help="local debugging only")
    s.add_argument("--no-reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    a = sub.add_parser("audit", help="read or verify the audit log")
    a.add_argument("--tail", type=int, default=20)
    a.add_argument("--verify", action="store_true")
    a.set_defaults(fn=cmd_audit)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
