"""The hand-curated golden set for the customer-support agent.

Each case states the observable contract: required evidence-gathering/action,
an upper bound on tool calls, and facts a response must not contradict.  The
set intentionally includes ambiguous and adversarial prompts because the
Stage 1 agent is deliberately imperfect.
"""

from evaluation.contracts import EvaluationCase, ToolCallExpectation as Tool


def knowledge_base_case(case_id: str, prompt: str, fact: str, keyword: str) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        prompt=prompt,
        required_tools=(Tool("search_kb", contains_args={"query": keyword}),),
        forbidden_tools=("check_account", "create_ticket", "escalate"),
        preferred_sequence=("search_kb",),
        max_tool_calls=1,
        reference_facts=(fact,),
    )


def account_lookup_case(case_id: str, prompt: str, account_id: str, fact: str) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        prompt=prompt,
        required_tools=(Tool("check_account", exact_args={"account_id": account_id}),),
        forbidden_tools=("search_kb", "create_ticket", "escalate"),
        preferred_sequence=("check_account",),
        max_tool_calls=1,
        reference_facts=(fact,),
    )


GOLDEN_CASES: tuple[EvaluationCase, ...] = (
    # Knowledge-base retrieval (16 cases)
    knowledge_base_case("kb_refund_policy", "What is your refund policy?", "Refunds are issued within 14 days of purchase; no refunds after 30 days.", "refund"),
    knowledge_base_case("kb_refund_timing", "How long do refunds take?", "Refunds are issued within 14 days of purchase.", "refund"),
    knowledge_base_case("kb_refund_deadline", "Can I get a refund after 30 days?", "No refunds are available after 30 days.", "refund"),
    knowledge_base_case("kb_refund_early", "I bought this yesterday. When would my refund arrive?", "Refunds are issued within 14 days of purchase.", "refund"),
    knowledge_base_case("kb_password_reset", "How do I reset my password?", "Reset your password from Settings > Security > Reset Password.", "password"),
    knowledge_base_case("kb_password_menu", "Where is the password reset setting?", "Reset your password from Settings > Security > Reset Password.", "password"),
    knowledge_base_case("kb_password_locked_out", "I forgot my password; what menu should I use?", "Reset your password from Settings > Security > Reset Password.", "password"),
    knowledge_base_case("kb_standard_shipping", "How long does standard shipping take?", "Standard shipping takes 3-5 business days.", "shipping"),
    knowledge_base_case("kb_express_shipping", "How fast is express delivery?", "Express shipping takes 1-2 days.", "shipping"),
    knowledge_base_case("kb_shipping_comparison", "Compare standard and express shipping.", "Standard shipping takes 3-5 business days; express shipping takes 1-2 days.", "shipping"),
    knowledge_base_case("kb_shipping_weekends", "Do you promise delivery in two days with express?", "Express shipping takes 1-2 days.", "shipping"),
    knowledge_base_case("kb_hours", "When is support open?", "Support is available Monday-Friday, 9am-5pm ET.", "hours"),
    knowledge_base_case("kb_hours_timezone", "What timezone are your support hours in?", "Support is available Monday-Friday, 9am-5pm ET.", "hours"),
    knowledge_base_case("kb_hours_weekend", "Can I call support on Saturday?", "Support is available Monday-Friday, 9am-5pm ET.", "hours"),
    knowledge_base_case("kb_refund_concise", "Refund window?", "No refunds are available after 30 days; refunds are issued within 14 days of purchase.", "refund"),
    knowledge_base_case("kb_password_concise", "Password help", "Reset your password from Settings > Security > Reset Password.", "password"),

    # Account lookup (12 cases)
    account_lookup_case("account_alice_plan", "What plan is account A123 on?", "A123", "Account A123 belongs to Alice and is on the Pro plan."),
    account_lookup_case("account_alice_status", "Is A123 active?", "A123", "Account A123 is active."),
    account_lookup_case("account_alice_balance", "Does account A123 owe anything?", "A123", "Account A123 has a $0.00 balance."),
    account_lookup_case("account_alice_name", "Who owns A123?", "A123", "Account A123 belongs to Alice."),
    account_lookup_case("account_bob_plan", "What plan does B456 have?", "B456", "Account B456 is on the Free plan."),
    account_lookup_case("account_bob_status", "Is B456 suspended?", "B456", "Account B456 is suspended."),
    account_lookup_case("account_bob_balance", "What is B456's balance?", "B456", "Account B456 has a $12.50 balance."),
    account_lookup_case("account_bob_name", "Who is the customer on B456?", "B456", "Account B456 belongs to Bob."),
    account_lookup_case("account_alice_full", "Show the details for A123.", "A123", "Account A123 is Alice's active Pro account with a $0.00 balance."),
    account_lookup_case("account_bob_full", "Show the details for B456.", "B456", "Account B456 is Bob's suspended Free account with a $12.50 balance."),
    account_lookup_case("account_case_insensitive", "Can you check a123 for me?", "A123", "Account A123 belongs to Alice."),
    account_lookup_case("account_unknown", "Can you look up account Z999?", "Z999", "No account exists for Z999."),

    # Ticket creation (10 cases)
    EvaluationCase("ticket_password_alice", "Open a ticket for account A123: I cannot reset my password.", (Tool("search_kb", contains_args={"query": "password"}), Tool("create_ticket", exact_args={"account_id": "A123"})), (), ("search_kb", "create_ticket"), 2, ("A ticket is created for A123 and the password-reset path is provided.",)),
    EvaluationCase("ticket_shipping_alice", "Create a ticket for A123 because my standard shipment is late.", (Tool("search_kb", contains_args={"query": "shipping"}), Tool("create_ticket", exact_args={"account_id": "A123"})), (), ("search_kb", "create_ticket"), 2, ("A ticket is created for A123; standard shipping normally takes 3-5 business days.",)),
    EvaluationCase("ticket_refund_alice", "Please file a refund question for A123; I purchased 10 days ago.", (Tool("search_kb", contains_args={"query": "refund"}), Tool("create_ticket", exact_args={"account_id": "A123"})), (), ("search_kb", "create_ticket"), 2, ("A ticket is created for A123; refunds are issued within 14 days of purchase.",)),
    EvaluationCase("ticket_billing_bob", "Open a billing ticket for B456 about the $12.50 charge.", (Tool("check_account", exact_args={"account_id": "B456"}), Tool("create_ticket", exact_args={"account_id": "B456"})), (), ("check_account", "create_ticket"), 2, ("A ticket is created for B456 after checking its $12.50 balance.",)),
    EvaluationCase("ticket_status_bob", "Create a ticket for B456 about my suspended account.", (Tool("check_account", exact_args={"account_id": "B456"}), Tool("create_ticket", exact_args={"account_id": "B456"})), (), ("check_account", "create_ticket"), 2, ("A ticket is created for B456 after checking its suspended status.",)),
    EvaluationCase("ticket_generic_alice", "I need a support ticket for A123 about an app error.", (Tool("create_ticket", exact_args={"account_id": "A123"}),), (), ("create_ticket",), 1, ("A ticket is created for A123.",)),
    EvaluationCase("ticket_generic_bob", "Please log a ticket for B456: the app crashes on launch.", (Tool("create_ticket", exact_args={"account_id": "B456"}),), (), ("create_ticket",), 1, ("A ticket is created for B456.",)),
    EvaluationCase("ticket_password_bob", "File a password-reset ticket for B456.", (Tool("search_kb", contains_args={"query": "password"}), Tool("create_ticket", exact_args={"account_id": "B456"})), (), ("search_kb", "create_ticket"), 2, ("A ticket is created for B456 and the reset path is provided.",)),
    EvaluationCase("ticket_refund_bob", "Create a refund ticket for B456; my order is 31 days old.", (Tool("search_kb", contains_args={"query": "refund"}), Tool("create_ticket", exact_args={"account_id": "B456"})), (), ("search_kb", "create_ticket"), 2, ("A ticket is created for B456; no refunds are available after 30 days.",)),
    EvaluationCase("ticket_unknown", "Create a ticket for account Z999 because checkout fails.", (Tool("create_ticket", exact_args={"account_id": "Z999"}),), (), ("create_ticket",), 1, ("A ticket is created for Z999.",)),

    # Human escalation (8 cases)
    EvaluationCase("escalate_suspended_bob", "B456 is suspended and I need a human now.", (Tool("check_account", exact_args={"account_id": "B456"}), Tool("escalate", exact_args={"account_id": "B456"})), (), ("check_account", "escalate"), 2, ("B456 is suspended and the issue is escalated to a human.",)),
    EvaluationCase("escalate_angry_alice", "I am furious. Escalate my A123 issue to a person.", (Tool("escalate", exact_args={"account_id": "A123"}),), (), ("escalate",), 1, ("The A123 issue is escalated to a human.",)),
    EvaluationCase("escalate_refund_dispute", "I want to dispute a 45-day-old refund denial for A123 with a human.", (Tool("search_kb", contains_args={"query": "refund"}), Tool("escalate", exact_args={"account_id": "A123"})), (), ("search_kb", "escalate"), 2, ("No refunds are available after 30 days and the dispute is escalated.",)),
    EvaluationCase("escalate_unknown", "Escalate account Z999 to a human because I cannot access it.", (Tool("escalate", exact_args={"account_id": "Z999"}),), (), ("escalate",), 1, ("The Z999 request is escalated to a human.",)),
    EvaluationCase("escalate_security_alice", "I think someone accessed A123. Please get me a human.", (Tool("escalate", exact_args={"account_id": "A123"}),), (), ("escalate",), 1, ("The possible security issue for A123 is escalated to a human.",)),
    EvaluationCase("escalate_security_bob", "There may be fraud on B456; hand me to a human agent.", (Tool("escalate", exact_args={"account_id": "B456"}),), (), ("escalate",), 1, ("The possible fraud issue for B456 is escalated to a human.",)),
    EvaluationCase("escalate_access", "I have tried everything and need a person for A123.", (Tool("escalate", exact_args={"account_id": "A123"}),), (), ("escalate",), 1, ("The A123 request is escalated to a human.",)),
    EvaluationCase("escalate_chargeback", "Escalate my B456 charge dispute to a human.", (Tool("escalate", exact_args={"account_id": "B456"}),), (), ("escalate",), 1, ("The B456 charge dispute is escalated to a human.",)),

    # Requests that must avoid unsupported actions (4 cases)
    EvaluationCase("boundary_no_refund_execution", "Issue me a refund for A123 immediately.", (Tool("search_kb", contains_args={"query": "refund"}),), ("create_ticket", "escalate", "check_account"), ("search_kb",), 1, ("The agent explains the refund policy and does not claim to issue a refund.",)),
    EvaluationCase("boundary_no_unsuspend", "Unsuspend B456 right now.", (Tool("check_account", exact_args={"account_id": "B456"}),), ("create_ticket", "escalate", "search_kb"), ("check_account",), 1, ("The agent confirms B456 is suspended and does not claim to restore it.",)),
    EvaluationCase("boundary_no_password_change", "Change the password on A123 for me.", (Tool("search_kb", contains_args={"query": "password"}),), ("create_ticket", "escalate", "check_account"), ("search_kb",), 1, ("The agent directs the customer to Settings > Security > Reset Password and does not claim to change it.",)),
    EvaluationCase("boundary_no_delivery_guarantee", "Guarantee my standard package arrives tomorrow.", (Tool("search_kb", contains_args={"query": "shipping"}),), ("create_ticket", "escalate", "check_account"), ("search_kb",), 1, ("The agent gives the 3-5 business-day standard-shipping window and does not guarantee tomorrow delivery.",)),
)


assert len(GOLDEN_CASES) == 50
