# Fleet AI

[![Checks](https://github.com/kimtour/Fleet-AI/actions/workflows/checks.yml/badge.svg)](https://github.com/kimtour/Fleet-AI/actions/workflows/checks.yml)

Evidence-grounded fleet health analysis and human-approved operations.

Fleet Copilot turns recorded device telemetry into cited fleet answers, time-series insights, and proposed IT actions. Python and LangGraph provide an explicit plan → tool → reflect loop; deterministic analytics and rendering keep fleet facts grounded in the dataset. A separate human command approves simulated actions.

Start with [QUICKSTART.md](QUICKSTART.md) for a short walkthrough. The repository includes source, a sample dataset, locked dependencies and verification reports. No manual dataset download is needed.

## Setup

Requirements: Python 3.10+; [uv](https://docs.astral.sh/uv/getting-started/installation/) for the locked LangGraph environment.

```bash
git clone https://github.com/kimtour/Fleet-AI.git
cd Fleet-AI
uv sync --locked --extra langgraph
uv run python -m fleet_copilot --backend langgraph --tenant acme-001 ask "Which devices at Acme are low on disk space?"
```

The deterministic native backend also runs without third-party packages:

```bash
python3 -m fleet_copilot --backend native --tenant acme-001 ask "Which devices at Acme are low on disk space?"
```

`--backend auto` is the default: LangGraph when installed, otherwise native. Both backends use the same node functions and have been tested locally. To build and install a wheel, run `uv build --wheel`, then `python -m pip install dist/fleet_copilot-1.0.0-py3-none-any.whl`; the `fleet-copilot` command then works outside the source directory using its packaged dataset. `--data` selects a different input file.

## Questions and insights

```bash
uv run python -m fleet_copilot --tenant acme-001 ask "Show laptops failing high-severity compliance checks."
uv run python -m fleet_copilot --tenant acme-001 ask "How many devices run an OS older than macOS 15?"
uv run python -m fleet_copilot --tenant acme-001 ask "Show battery trends over time"
uv run python -m fleet_copilot --tenant acme-001 ask "Show screen lock failures over time"
uv run python -m fleet_copilot --tenant acme-001 ask "Show disks below 20 percent free" --json
```

The rule planner supports a bounded vocabulary of fleet questions. Supported filters retain their named check, severity, form factor, device, threshold, and capacity. Unsupported date windows, comparison operators, or ambiguous action requests are refused or request clarification. The default low-disk threshold is **less than 10% free**. Custom read thresholds support strict-below percentages; both `%` and `percent` are accepted.

The bundled sample dataset contains 750 snapshots, 25 devices, three companies, and 14 distinct employee IDs, covering May 14–June 12, 2026. “Latest” refers to recorded snapshots, not present-day fleet state. There are no failed high-severity checks in this input, so that example returns an evidenced zero. Missing readings are treated as data gaps, not proof of health.

## Propose, review, and approve

```bash
uv run python -m fleet_copilot --tenant acme-001 ask "Propose actions for low disk space"
uv run python -m fleet_copilot --tenant acme-001 proposals
```

Review the printed arguments, evidence, and proposal ID. Approval is a separate command requiring `--role approver` and the exact confirmation text `APPROVE` followed by that actual ID. Natural-language chat and the REPL cannot approve actions.

Proposals expire after 24 hours. Re-proposing expired or evidence-invalidated work generates a fresh review ID and preserves the old record. Approval rechecks tenant, role, audited payload, expiry, dataset hashes, audit integrity, and current policy evidence. Replayed approvals and changes to executed proposals are refused.

All executions are local simulations. State is stored in `state/fleet.db`. No orders, employee messages, or external tickets are sent.

## Architecture

```text
scope_guard → plan → act → reflect ──→ act
     │                       │          ↑
     │                       └──────────┘
     └──→ synthesize ←────────┘
               │
             verify → answer

pending proposal → separate human command → revalidation
                 → atomic approval + simulated action + audit commit
```

- **Scope guard:** checks tenant references, device targets, destructive operations, and attempts to bypass approval before tool use.
- **Plan:** creates schema-validated tool steps and assumptions. The default planner is deterministic; an optional model adapter can propose read-tool plans.
- **Act:** executes one registered, tenant-bound tool and logs its result digest and evidence IDs.
- **Reflect:** expands observations into evidence-backed proposals, or tries one policy-supported alternative after an upgrade refusal. Execution is bounded to 40 tool steps.
- **Synthesize and verify:** render claims from structured observations, attach evidence, and withhold unsupported or foreign-tenant data claims.

The graph records observable plans, tool results, and policy decisions; it does not expose hidden model reasoning.

## Tool catalog

| Tool | Purpose |
|---|---|
| `fleet_overview(form_factor?)` | Counts, OS/form-factor breakdown, scoped data-quality notes |
| `find_low_disk(threshold_pct, form_factor?)` | Latest free-disk percentage below a threshold |
| `find_compliance_failures(severity?, check_id?, form_factor?)` | Failed checks and persistence |
| `count_os_below(product, major)` | Compare recorded OS major versions |
| `get_device_history(device_id)` | Device telemetry profile over the recorded window |
| `detect_insights(kind, check_id?, form_factor?, device_id?, severity?)` | Storage, RAM, battery, and compliance trends |
| `create_upgrade_order(device_id, component, spec)` | Propose a capacity sizing review |
| `open_remediation_ticket(device_id, check_id, note)` | Propose a ticket for an evidenced failure |
| `flag_device_for_replacement(device_id, reason, basis?)` | Propose a battery or storage replacement review |
| `notify_employee(employee_id, message)` | Propose a topic-scoped employee notification |

Action tools only create pending proposals. There is no approve/execute tool in the agent registry. No tool accepts a tenant selector; caller context supplies tenant scope.

## Grounding and analytics

Evidence records contain device identity, snapshot time or range, relevant fields, and the producing tool. Answers cite evidence IDs and include a legend; JSON output also carries the dataset SHA-256. Coverage evidence supports aggregate and zero-result answers. Unreported compliance observations are unknown rather than passing.

Insights use explainable statistics: storage slope and fit, RAM utilization share/mean/p95, relative battery capacity change and service flags, and weekly compliance counts with pass-to-fail transitions. The dataset lacks design battery capacity, so absolute battery health and remaining life are not inferred. Snapshots cannot prove continuous utilization or causation.

Default policies are explicit in `analytics.Thresholds`. RAM consistency means more than 85% utilization in at least 70% of snapshots. Tickets require at least two consecutive failed observations. Battery replacement review requires at least seven service-recommended days with declining capacity; a separate criterion covers exhausted, non-upgradable storage.

## Safety and persistence

Tenant-bound views constrain reads. Tool schemas reject unknown arguments, and the host validates model-proposed device targets before executing tools. Action eligibility is independently recomputed from source data; a supplied “allowed” decision is insufficient.

SQLite `BEGIN IMMEDIATE` transactions coordinate processes. Approval, outbox insertion, proposal status, and hash-chained audit events commit together or roll back together. Tests terminate a process after the action insert and verify that no partial approval or action survives. A unique idempotency key prevents duplicate execution.

Hash chaining detects edited records and interior deletion. Detecting complete database rewriting or tail truncation needs an external trusted anchor. Operator-controlled CLI identity flags are a local trust boundary, not authentication. A hosted service must derive identity and tenant from authenticated sessions.

Legacy JSON state directories are refused rather than silently imported into an unverifiable history. Use a fresh `--state` directory for this release; legacy files are preserved.

## Evaluations and verification

```bash
uv run python -m unittest discover -s tests
uv run python -m eval.run_eval
uv run python -m eval.mutation_check
uv build --wheel
```

Verified locally on macOS/Python 3.14.2:

- **68 tests pass** with native and LangGraph execution, including crash recovery, concurrent approval, expiry, tampering, tenant isolation, intent/filter handling, and mocked provider contracts.
- **61 supplied-data evaluation cases pass.** Grounding checks resolve 126 evidence records against source data; 272 evaluated data claims have valid same-tenant citations.
- **Six injected faults are caught**, with a passing baseline required before and after mutation tests.
- **Eight concurrent processes across three tenants:** 19 approvals, 19 unique simulated executions, valid audit chain.
- The wheel installs in a clean environment and runs outside the source tree with its bundled dataset.

Reports are in `verification/`. Synthetic tests cover boundaries and adversarial cases not represented by the supplied input. The evaluation labels were computed separately from the agent, but the questions and policies remain authored within this project; passing scores do not establish arbitrary-language robustness.

## Model adapter and limits

The optional adapter requires `ANTHROPIC_API_KEY` and `FLEET_COPILOT_MODEL`, then `--planner llm`. Only questions, policy instructions, and the read-tool catalog are sent to the provider. Telemetry stays local. Missing configuration, timeouts, invalid JSON, or attempted action tools fall back to deterministic planning. Model-selected parameters remain visible in the answer and assumptions.

**Live provider calls remain unverified.** Mocked HTTP tests check the adapter contract; they do not measure live model quality. GitHub Actions runs native checks on Linux, macOS, and Windows, plus a LangGraph job. The badge above links to the current results.

Hardware compatibility is an explicit external policy assumption, not a telemetry fact. Upgrade specifications are sizing reviews rather than vendor SKUs. A real execution adapter needs independent hardware validation, fresh telemetry, authenticated approvals, and external idempotency. Installed-software risk scoring, real operational integrations, and persistent conversational memory are outside the current implementation.

## Repository contents

`fleet_copilot/` contains the graph, planners, analytics, tool registry, policy, persistence, and rendering. `tests/` contains regression tests; `eval/` contains independent labels, evaluation, and mutation tooling. `uv.lock` records dependencies. See [QUICKSTART.md](QUICKSTART.md) for setup and [verification/](verification/) for recorded checks.
