"""Offline stand-ins for the paid models, used in mock mode.

The mock extractor is a Pydantic AI `FunctionModel`, so mock mode still runs
the real agent, the real schema validation and the real grounding checks.
Only the "model" is swapped for simple rules.
"""

import re

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

NAME = re.compile(
    r"(?i:\bmy name is|\bthis is|\bit's|\bi'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)
PHONE = re.compile(r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]\d{4}\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
ADDRESS = re.compile(
    r"\b\d{1,6}\s+[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+"
    r"(?:Lane|Ln|Drive|Dr|Road|Rd|Street|St|Avenue|Ave|Court|Ct|Way|Blvd|Circle|Cir)\b\.?"
    r"(?:,\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)?"
    r"(?:,\s*[A-Z]{2})?(?:\s+\d{5})?"
)

SERVICE_KEYWORDS = {
    "repair": r"stopped|not cooling|not heating|no heat|broken|leak|short cycling|be fixed|repair|not working",
    "maintenance": r"tune-?up|maintenance|check-?up",
    "replacement": r"new one|replace|replacement|new system",
    "installation": r"\binstall",
    "inspection": r"inspect",
}
SYSTEM_KEYWORDS = {
    "central_ac": r"\bAC\b|air condition",
    "heat_pump": r"heat pump",
    "furnace": r"furnace",
    "boiler": r"boiler",
    "mini_split": r"mini[- ]split",
    "thermostat": r"thermostat",
}
CALLBACK_HINTS = re.compile(
    r"reach me|call|text|email me|best time|mornings?|evenings?|afternoons?|contact",
    re.IGNORECASE,
)
DECLINE = re.compile(
    r"(?:don't|do not|no need to)\s+(?:call|contact|email|text|follow up)|not interested|no follow[- ]?up",
    re.IGNORECASE,
)
URGENT = re.compile(r"emergency|urgent|asap|right away", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def rule_based_extract(inquiry: str) -> dict:
    """Pull lead fields out of an inquiry with plain rules. Unknown stays None."""
    sentences = _sentences(inquiry)

    def first(pattern: re.Pattern) -> str | None:
        match = pattern.search(inquiry)
        return match.group(match.lastindex or 0) if match else None

    services = [
        name for name, pattern in SERVICE_KEYWORDS.items()
        if re.search(pattern, inquiry, re.IGNORECASE)
    ]
    systems = [
        name for name, pattern in SYSTEM_KEYWORDS.items()
        if re.search(pattern, inquiry, re.IGNORECASE)
    ]

    callback = next((s for s in sentences if CALLBACK_HINTS.search(s)), None)
    issue = next(
        (s for s in sentences if any(re.search(p, s, re.IGNORECASE) for p in SERVICE_KEYWORDS.values())),
        None,
    )

    return {
        "intent": "hvac_service" if services or systems else "out_of_scope",
        "customer_name": first(NAME),
        "phone": first(PHONE),
        "email": first(EMAIL),
        "service_address": first(ADDRESS),
        "service_type": services[0] if len(services) == 1 else None,
        "service_type_candidates": services if len(services) > 1 else [],
        "system_type": systems[0] if len(systems) == 1 else None,
        "urgency": "emergency" if URGENT.search(inquiry) else None,
        "issue_quote": issue,
        "callback_preference": callback,
        "declined_follow_up": bool(DECLINE.search(inquiry)),
    }


def _mock_extractor(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    prompts = [
        part.content for message in messages for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]
    return ModelResponse(
        parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=rule_based_extract(prompts[0]))]
    )


def mock_extraction_model() -> FunctionModel:
    return FunctionModel(_mock_extractor, model_name="mock-rule-extractor")
