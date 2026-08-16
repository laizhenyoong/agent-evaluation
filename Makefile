PYTHON := .venv/bin/python

.PHONY: test capture judge report baseline regression

test:
	$(PYTHON) -m unittest discover -s tests -v

# Capture a run and score it against the committed baseline.
capture:
	$(PYTHON) -m evaluation.capture --output results/latest.jsonl

judge:
	$(PYTHON) -m evaluation.judge --traces results/latest.jsonl --output results/judgements.jsonl

report:
	$(PYTHON) -m evaluation.report --traces results/latest.jsonl \
		--baseline results/baseline-summary.json --tolerance 0.02

# Re-cut the baseline. Only run this when a change is a deliberate improvement.
baseline:
	$(PYTHON) -m evaluation.capture --output results/baseline.jsonl
	$(PYTHON) -m evaluation.report --traces results/baseline.jsonl \
		--write-baseline results/baseline-summary.json

# Stage 4 demo: a "helpful" prompt tweak that quietly costs trajectory quality.
regression:
	$(PYTHON) -m evaluation.capture --output results/regression.jsonl \
		--system-prompt "You are a customer support agent. Be thorough: check the \
customer's account and search the knowledge base before answering anything."
	$(PYTHON) -m evaluation.report --traces results/regression.jsonl \
		--baseline results/baseline-summary.json --tolerance 0.02
