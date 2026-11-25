"""Validated lead schema for the HVAC lead-intake demo.

Follows the FleetMind AI extraction conventions:
- one Pydantic model per request type, with `intent` first
- every field the customer did not give is `None` (never invented)
- allowed values live in module-level tuples / Literals
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SERVICE_TYPES = ("repair", "maintenance", "installation", "replacement", "inspection")
SYSTEM_TYPES = (
    "central_ac", "heat_pump", "furnace", "boiler",
    "mini_split", "thermostat", "ductwork",
)
URGENCY_LEVELS = ("emergency", "soon", "routine")

ServiceType = Literal["repair", "maintenance", "installation", "replacement", "inspection"]
SystemType = Literal[
    "central_ac", "heat_pump", "furnace", "boiler",
    "mini_split", "thermostat", "ductwork",
]
Urgency = Literal["emergency", "soon", "routine"]

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


class HVACLead(BaseModel):
    """Lead fields extracted from one customer inquiry."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["hvac_service", "out_of_scope"]
    customer_name: str | None = None
    phone: str | None = Field(
        default=None, description="Phone number exactly as the customer wrote it."
    )
    email: str | None = None
    service_address: str | None = None
    service_type: ServiceType | None = Field(
        default=None,
        description="Set only when the customer clearly asked for one type of service.",
    )
    service_type_candidates: list[ServiceType] = Field(
        default_factory=list,
        description="When the request is ambiguous, the service types it could be.",
    )
    system_type: SystemType | None = None
    urgency: Urgency | None = Field(
        default=None, description="Only when the customer states how urgent it is."
    )
    issue_quote: str | None = Field(
        default=None,
        description="The customer's own sentence describing the problem, copied word for word.",
    )
    callback_preference: str | None = Field(
        default=None,
        description=(
            "The customer's own words about how or when to contact them "
            "(including a request not to be contacted), copied word for word."
        ),
    )
    declined_follow_up: bool = False

    @field_validator("phone")
    @classmethod
    def phone_has_ten_digits(cls, value: str | None) -> str | None:
        if value is None:
            return value
        digits = re.sub(r"\D", "", value)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10:
            raise ValueError("Phone number must have 10 digits.")
        return value

    @field_validator("email")
    @classmethod
    def email_looks_valid(cls, value: str | None) -> str | None:
        if value is not None and not EMAIL_PATTERN.match(value):
            raise ValueError("Email address is not valid.")
        return value

    @field_validator(
        "customer_name", "phone", "email", "service_address",
        "issue_quote", "callback_preference",
        mode="before",
    )
    @classmethod
    def blank_means_unknown(cls, value: str | None) -> str | None:
        # An empty string is not a value; store it as unknown.
        if value is not None and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> "HVACLead":
        if self.service_type is not None and self.service_type_candidates:
            raise ValueError(
                "Use service_type_candidates only when service_type is unknown."
            )
        if self.declined_follow_up and self.callback_preference is None:
            raise ValueError(
                "A declined follow-up must keep the customer's words in callback_preference."
            )
        return self


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def grounding_problems(lead: HVACLead, inquiry: str) -> list[str]:
    """Return fields that are not supported by the inquiry text.

    Quoted fields must appear word for word. Contact details must appear
    in the inquiry, so the model cannot invent them.
    """
    problems: list[str] = []
    source = _normalize_space(inquiry)
    source_digits = re.sub(r"\D", "", inquiry)

    for field in ("callback_preference", "issue_quote", "customer_name", "email"):
        value = getattr(lead, field)
        if value is not None and _normalize_space(value) not in source:
            problems.append(f"{field} must be copied word for word from the inquiry")

    if lead.phone is not None:
        digits = re.sub(r"\D", "", lead.phone)
        if digits[-10:] not in source_digits:
            problems.append("phone does not appear in the inquiry")

    return problems


class FollowUpDraft(BaseModel):
    """A follow-up message for the contractor to review. Never sent automatically."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["pending_contractor_review", "not_drafted"]
    channel: Literal["sms", "email", "call_script"] | None = None
    subject: str | None = None
    body: str | None = None
    review_flags: list[str] = Field(default_factory=list)
    reason_not_drafted: str | None = None
    outbound_sent: Literal[False] = False


class LeadReview(BaseModel):
    """Everything the contractor sees for one inquiry."""

    mode: Literal["mock", "live"]
    inquiry: str
    lead: HVACLead
    follow_up: FollowUpDraft
