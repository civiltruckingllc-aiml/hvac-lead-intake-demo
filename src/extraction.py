"""Pydantic AI lead extraction for the HVAC lead-intake demo."""

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent, ModelRetry, RunContext  # noqa: E402

from .config import Mode, live_model_id  # noqa: E402
from .mock_models import mock_extraction_model  # noqa: E402
from .schemas import SERVICE_TYPES, SYSTEM_TYPES, URGENCY_LEVELS, HVACLead, grounding_problems  # noqa: E402

INSTRUCTIONS = f"""
You extract lead details from a homeowner's HVAC service inquiry.

Fill the output fields. Use null for anything the customer did not say.
Never invent or guess names, phone numbers, emails, addresses, or urgency.

Set intent to "hvac_service" for heating or cooling service requests.
Set intent to "out_of_scope" for anything else.

Allowed service types: {list(SERVICE_TYPES)}
Allowed system types: {list(SYSTEM_TYPES)}
Allowed urgency levels: {list(URGENCY_LEVELS)}

If the customer is unsure which service they need (for example repair OR
replace), set service_type to null and list the options in
service_type_candidates.

Copy these fields word for word from the inquiry. Do not rephrase,
shorten, fix spelling, or change punctuation:
- callback_preference: the customer's sentence about how or when to
  contact them, including any request NOT to be contacted.
- issue_quote: the customer's sentence describing the problem.
- phone and email: exactly as written.

Set declined_follow_up to true only if the customer says not to contact
them or that they do not want a follow-up.

Do not write a reply to the customer. Only extract.
"""


def build_agent(mode: Mode) -> Agent[str, HVACLead]:
    model = mock_extraction_model() if mode == "mock" else f"anthropic:{live_model_id()}"
    agent = Agent(
        model,
        output_type=HVACLead,
        instructions=INSTRUCTIONS,
        deps_type=str,
        retries=2,
        model_settings={"temperature": 0},
    )

    @agent.output_validator
    def must_match_inquiry(ctx: RunContext[str], lead: HVACLead) -> HVACLead:
        problems = grounding_problems(lead, ctx.deps)
        if problems:
            raise ModelRetry("Fix these fields: " + "; ".join(problems))
        return lead

    return agent


def extract_lead(inquiry: str, mode: Mode = "mock") -> HVACLead:
    """Turn one inquiry into validated lead fields."""
    if not inquiry.strip():
        raise ValueError("Please enter a customer inquiry.")
    result = build_agent(mode).run_sync(inquiry, deps=inquiry)
    return result.output
