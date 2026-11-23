# Fleet AI quickstart

The core runs offline with Python 3.10+. Install uv for a locked environment with LangGraph.

```bash
uv sync --locked --extra langgraph
uv run python -m fleet_copilot --tenant acme-001 ask "Which devices are low on disk space?"
uv run python -m fleet_copilot --tenant acme-001 ask "Show screen lock failures over time"
uv run python -m fleet_copilot --tenant acme-001 ask "Show battery trends over time"
```

Add `--json` after the question for structured evidence and the dataset hash. The sample data is bundled with both the source and the built wheel.

## Review proposed actions

```bash
uv run python -m fleet_copilot --tenant acme-001 ask "Propose actions for low disk space"
uv run python -m fleet_copilot --tenant acme-001 proposals
```

Inspect a printed proposal ID and its arguments. Approval uses a separate command with `--role approver`, `approve`, that ID, and `--confirm "APPROVE <actual ID>"`. The Python API trusts its caller context; CLI flags do not implement authentication. All action execution is simulated locally in `state/fleet.db`.

## Run checks and build

```bash
uv run python -m unittest discover -s tests
uv run python -m eval.run_eval
uv run python -m eval.mutation_check
uv build --wheel
```

Without dependencies, use `python3 -m fleet_copilot --backend native` with the same arguments. The default backend chooses LangGraph when installed, otherwise native.

See the README for architecture, tool schemas, evidence policies, supported question forms and operational limits. Live model calls require provider credentials and are optional. Recorded local checks are under `verification/`; GitHub Actions runs cross-platform checks on pushes and pull requests.
