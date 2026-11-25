"""Follow-up drafting with the Claude API, for contractor review only."""

import json
import re

import anthropic

from .config import Mode, live_model_id
from .schemas import FollowUpDraft, HVACLead

SYSTEM_PROMPT = """
You draft a short follow-up message from an HVAC contractor to a homeowner
who sent an inquiry. A person at the contractor will review and edit your
draft before anything is sent.

Rules:
- Use only the lead details provided. Do not invent names, dates,
  appointment times, prices, or diagnoses.
- Respect the customer's callback preference. It is quoted exactly; follow it.
- If service_type is null and service_type_candidates is not empty, ask one
  simple question about which service they want.
- Do not promise a price, a time slot, or that the problem can be fixed.
- Use the placeholders [Your Name] and [Company Name] for the sender.
- Keep it under 110 words, friendly and plain.
- For channel "call_script", write short talking points for the contractor's
  phone call instead of a message.
Return only the message text.
"""

SERVICE_WORDS = {
    "repair": "repair",
    "maintenance": "tune-up",
    "installation": "installation",
    "replacement": "replacement",
    "inspection": "inspection",
}
SYSTEM_WORDS = {
    "central_ac": "AC",
    "heat_pump": "heat pump",
    "furnace": "furnace",
    "boiler": "boiler",
    "mini_split": "mini-split",
    "thermostat": "thermostat",
    "ductwork": "ductwork",
}


def review_flags(lead: HVACLead) -> list[str]:
    """Things the contractor should check before replying."""
    flags = []
    if lead.declined_follow_up:
        flags.append("customer_declined_follow_up")
    if lead.phone is None:
        flags.append("missing_phone")
    if lead.email is None:
        flags.append("missing_email")
    if lead.phone is None and lead.email is None:
        flags.append("no_contact_method")
    if lead.service_address is None:
        flags.append("missing_service_address")
    if lead.service_type is None:
        flags.append("ambiguous_service_type" if lead.service_type_candidates else "missing_service_type")
    if lead.urgency == "emergency":
        flags.append("customer_says_emergency")
    if lead.intent == "out_of_scope":
        flags.append("out_of_scope")
    return flags


def pick_channel(lead: HVACLead) -> str | None:
    """Choose how to reach the customer, following their own words first."""
    pref = (lead.callback_preference or "").lower()
    if lead.phone and re.search(r"\btext", pref):
        return "sms"
    if lead.email and "email" in pref:
        return "email"
    if lead.phone:
        return "call_script"
    if lead.email:
        return "email"
    return None


def _first_name(lead: HVACLead) -> str:
    return lead.customer_name.split()[0] if lead.customer_name else "there"


def mock_draft_body(lead: HVACLead, channel: str) -> str:
    """Template draft so the demo runs with no paid calls."""
    system = SYSTEM_WORDS.get(lead.system_type or "", "HVAC system")
    service = SERVICE_WORDS.get(lead.service_type or "", "service")
    if lead.service_type_candidates:
        options = " or ".join(SERVICE_WORDS[s] for s in lead.service_type_candidates)
        ask = f"Are you leaning toward a {options}, or would you like a technician to take a look first?"
    else:
        ask = f"We'd be glad to set up a {service} visit."

    if channel == "call_script":
        return "\n".join([
            f"- Greet {_first_name(lead)}; you're calling from [Company Name] about their {system}.",
            f"- Customer's words on contact: \"{lead.callback_preference}\"" if lead.callback_preference else "- No callback preference given.",
            f"- Ask: \"{ask}\"" if lead.service_type_candidates else f"- Offer: \"{ask}\"",
            "- Confirm the service address and a visit window. Do not quote a price on the call.",
        ])

    lines = [
        f"Hi {_first_name(lead)}, this is [Your Name] with [Company Name].",
        f"Thanks for reaching out about your {system}. {ask}",
        "What day and time window works best for you?",
    ]
    if channel == "email":
        lines.append("Thanks,\n[Your Name]\n[Company Name]")
    return "\n\n".join(lines) if channel == "email" else " ".join(lines)


def live_draft_body(lead: HVACLead, channel: str, flags: list[str], client=None) -> str:
    client = client or anthropic.Anthropic(timeout=30.0, max_retries=1)
    payload = {"channel": channel, "lead": lead.model_dump(), "review_flags": flags}
    response = client.messages.create(
        model=live_model_id(),
        max_tokens=500,
        temperature=0.2,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ValueError("Claude returned an empty draft.")
    return text


def draft_follow_up(lead: HVACLead, mode: Mode = "mock", client=None) -> FollowUpDraft:
    """Draft a follow-up for contractor review. Never sends anything."""
    flags = review_flags(lead)

    if lead.declined_follow_up:
        return FollowUpDraft(
            status="not_drafted",
            review_flags=flags,
            reason_not_drafted="Customer asked not to be contacted. See callback_preference for their exact words.",
        )
    if lead.intent == "out_of_scope":
        return FollowUpDraft(status="not_drafted", review_flags=flags, reason_not_drafted="Not an HVAC service request.")

    channel = pick_channel(lead)
    if channel is None:
        return FollowUpDraft(
            status="not_drafted",
            review_flags=flags,
            reason_not_drafted="No phone number or email to reply to.",
        )

    if mode == "mock":
        body = mock_draft_body(lead, channel)
    else:
        body = live_draft_body(lead, channel, flags, client=client)

    if "$" in body:
        flags.append("draft_mentions_price")

    return FollowUpDraft(
        status="pending_contractor_review",
        channel=channel,
        subject="Your HVAC service request" if channel == "email" else None,
        body=body,
        review_flags=flags,
    )
