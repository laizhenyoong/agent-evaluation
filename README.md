# Agent Eval Lab

A closed evaluation loop around one deliberately-imperfect tool-using support
agent: trace it, score its *trajectories* rather than just its answers, and
gate changes against a baseline.

## Setup

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Capture needs [Ollama](https://ollama.com) running locally with `qwen3:30b`.
Scoring does not.

## Commands

```sh
make capture   # run the 50 golden cases -> results/latest.jsonl
make report    # score them, gate against the committed baseline
make judge     # optional: grade stored answers with a local LLM judge
make test      # offline scorer tests, no model required
make baseline  # re-cut the baseline (only for deliberate improvements)
```

## How it fits together

Capturing a trajectory and scoring it are separate steps, joined by a JSONL
trace file:

| Step | Module | Cost | Output |
|---|---|---|---|
| Capture | `evaluation.capture` | slow, needs Ollama | traces |
| Score | `evaluation.report` | free | summary + regression gate |
| Judge | `evaluation.judge` | slow, needs Ollama | judgements |
| Agreement | `evaluation.agreement` | free | per-label kappa |

Every trace carries the `agent_config` that produced it, so re-scoring a run,
re-judging it with a different prompt, or mining it for failures costs nothing
and re-runs nothing. It is also what makes CI possible: a hosted runner has no
Ollama, but it can re-score committed traces.

## Scoring

Three kinds of scorer, deliberately kept apart:

- **Deterministic** — required calls and arguments, forbidden calls, tool
  budget, retry loops.
- **Trajectory** — was the *path* economical and in the intended order?
- **LLM judge** — correctness, faithfulness to reference facts, tone. Opt-in,
  and it reads stored answers rather than live runs.

Scores are **graded, not binary**. A binary scorer cannot tell "took one
wasteful detour" from "did something completely different", so a prompt change
that halves the wasted calls would look identical to one that changed nothing.
`passed` is still reported for the gate, as the strict case `score == 1.0`.

The gate compares against a committed baseline rather than demanding a perfect
score — the agent is intentionally imperfect, so "100% or fail" would never go
green and would therefore gate nothing. `make regression` demonstrates it: a
"be thorough, always check the account first" prompt reads like an improvement
while quietly pushing the agent into unnecessary calls, and the trajectory
scorer catches what an answer-only eval would miss.

### Is the judge any good?

Hand-label some cases into a judgement file, then compare:

```sh
.venv/bin/python -m evaluation.agreement \
    results/judgements.jsonl labels/mine.jsonl --show-disagreements factually_correct
```

This reports raw agreement *and* Cohen's kappa per label. Raw agreement
flatters a judge whenever one verdict dominates: if 95 of 100 answers are
correct, two judges that always say "correct" agree 95% of the time while
sharing no judgement at all. Kappa subtracts that expected coincidence.

## The golden set

50 hand-curated cases in `evaluation/golden_dataset.py`, covering knowledge-base
retrieval, account lookup, ticket creation, escalation, and requests the agent
must refuse to act on. Each states an observable contract: required calls and
arguments, forbidden calls, a tool budget, a preferred order, and reference
facts.

Contracts live in `evaluation/contracts.py`; what a run actually did lives in
`evaluation/traces.py`. Keeping intent separate from observation is why a
scorer can never quietly grade a run against itself.

## Tracing

Phoenix is the human-facing debugging UI, receiving Strands' native
OpenTelemetry spans. It is not the eval substrate — the JSONL traces are.

```sh
PHOENIX_WORKING_DIR="$PWD/.phoenix" phoenix serve   # terminal 1
python support_agent.py                             # terminal 2
```

Open http://localhost:6006 and select the `customer-support-agent` project.
Defaults suit a local setup; override with `PHOENIX_COLLECTOR_ENDPOINT`,
`PHOENIX_PROJECT_NAME`, `OLLAMA_HOST`, or `EVAL_JUDGE_MODEL`.

Tracing intentionally records the fake account IDs and support requests in this
demo. Redact customer data before using this configuration with real traffic.
