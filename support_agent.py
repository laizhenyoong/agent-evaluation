"""Stages 1-2: a deliberately-unpolished, traced customer-support agent.

Goal is NOT a perfect agent. We want real trajectories (tool-call sequences)
with failure modes that later stages can measure. Runs locally on Ollama.

  Tools: search_kb, check_account, create_ticket, escalate
  Model: qwen3:30b via Ollama (tool-capable, local, free)
"""

from strands import Agent, tool
from strands.models.ollama import OllamaModel
from strands.telemetry import StrandsTelemetry
import atexit
import os

# --- Fake data ------------------------------------------------------------

KB = {
    "refunds": "Refunds are issued within 14 days of purchase. No refunds after 30 days.",
    "password reset": "Reset your password from Settings > Security > Reset Password.",
    "shipping": "Standard shipping takes 3-5 business days. Express takes 1-2 days.",
    "hours": "Support is available Monday-Friday, 9am-5pm ET.",
}

ACCOUNTS = {
    "A123": {"name": "Alice", "plan": "Pro", "status": "active", "balance": 0.0},
    "B456": {"name": "Bob", "plan": "Free", "status": "suspended", "balance": 12.50},
}

TICKETS = []      # filled by create_ticket
ESCALATIONS = []  # filled by escalate


def reset_demo_state() -> None:
    """Clear mutable fake-service state before an independent evaluation run."""
    TICKETS.clear()
    ESCALATIONS.clear()


# --- Tracing --------------------------------------------------------------

_telemetry: StrandsTelemetry | None = None


def configure_tracing() -> None:
    """Export Strands' built-in OpenTelemetry spans to a local Phoenix server."""
    global _telemetry

    if _telemetry is not None:
        return

    collector_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    trace_endpoint = collector_endpoint.rstrip("/")
    if not trace_endpoint.endswith("/v1/traces"):
        trace_endpoint = f"{trace_endpoint}/v1/traces"

    project_name = os.environ.get("PHOENIX_PROJECT_NAME", "customer-support-agent")
    os.environ.setdefault("OTEL_SERVICE_NAME", "customer-support-agent")

    _telemetry = StrandsTelemetry().setup_otlp_exporter(
        endpoint=trace_endpoint,
        headers={"x-project-name": project_name},
    )
    atexit.register(_telemetry.tracer_provider.shutdown)


# --- Tools ----------------------------------------------------------------

@tool
def search_kb(query: str) -> str:
    """Search the support knowledge base for an answer.

    Args:
        query: What the customer is asking about.
    """
    q = query.lower()
    hits = [text for key, text in KB.items() if key in q or any(w in q for w in key.split())]
    return "\n".join(hits) if hits else "No knowledge base article found."


@tool
def check_account(account_id: str) -> str:
    """Look up a customer's account details by account ID.

    Args:
        account_id: The customer's account ID, e.g. "A123".
    """
    acct = ACCOUNTS.get(account_id)
    if not acct:
        return f"No account found for {account_id}."
    return (f"Account {account_id}: {acct['name']}, plan={acct['plan']}, "
            f"status={acct['status']}, balance=${acct['balance']:.2f}")


@tool
def create_ticket(account_id: str, subject: str, description: str) -> str:
    """File a support ticket for a customer.

    Args:
        account_id: The customer's account ID.
        subject: Short summary of the issue.
        description: Full description of the issue.
    """
    ticket_id = f"T{1000 + len(TICKETS)}"
    TICKETS.append({"id": ticket_id, "account_id": account_id,
                    "subject": subject, "description": description})
    return f"Ticket {ticket_id} created."


@tool
def escalate(account_id: str, reason: str) -> str:
    """Hand the conversation off to a human support agent.

    Args:
        account_id: The customer's account ID.
        reason: Why this needs a human.
    """
    ESCALATIONS.append({"account_id": account_id, "reason": reason})
    return "Escalated to a human agent."


# --- Agent ----------------------------------------------------------------

# Deliberately minimal prompt: we want to see what the model does on its own.
SYSTEM_PROMPT = "You are a customer support agent. Help the customer using your tools."

DEFAULT_MODEL_ID = "qwen3:30b"


def build_agent(model_id: str = DEFAULT_MODEL_ID, system_prompt: str = SYSTEM_PROMPT) -> Agent:
    """Build the agent, letting an evaluation run vary the prompt or model.

    Both knobs are recorded on every trace, so a pass-rate change is traceable
    to the exact configuration that produced it.
    """
    configure_tracing()
    model = OllamaModel(host="http://localhost:11434", model_id=model_id)
    # callback_handler=None: suppress token streaming so we can print a clean trajectory.
    return Agent(
        model=model,
        tools=[search_kb, check_account, create_ticket, escalate],
        system_prompt=system_prompt,
        callback_handler=None,
    )


def print_trajectory(agent: Agent, query: str) -> None:
    """Run one query on a fresh agent and print its trajectory."""
    print(f"\n{'=' * 70}\nUSER: {query}\n{'-' * 70}")
    result = agent(query)
    for msg in agent.messages:
        for block in msg["content"]:
            if "toolUse" in block:
                tu = block["toolUse"]
                print(f"  [tool call] {tu['name']}({tu['input']})")
            elif "toolResult" in block:
                content = block["toolResult"]["content"][0].get("text", "")
                print(f"  [tool result] {content}")
    print(f"AGENT: {result}")


if __name__ == "__main__":
    # A few representative queries. Fresh agent per query so trajectories don't bleed.
    queries = [
        "What's your refund policy?",                       # simple KB lookup
        "What plan is account A123 on?",                    # needs check_account
        "My account B456 is suspended and I'm really angry, fix it now!",  # judgment call
        "I can't reset my password and I need a ticket opened for account A123.",  # multi-tool
    ]
    for q in queries:
        print_trajectory(build_agent(), q)

    print(f"\n{'=' * 70}\nTickets filed: {TICKETS}\nEscalations: {ESCALATIONS}")
