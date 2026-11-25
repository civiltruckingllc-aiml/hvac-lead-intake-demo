"""Tests for the HVAC lead-intake demo. All run offline (mock mode / fake clients)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from src import config, drafting, outbound
from src.extraction import build_agent, extract_lead
from src.mock_models import rule_based_extract
from src.pipeline import process_inquiry
from src.schemas import HVACLead, grounding_problems

INQUIRIES = Path(__file__).parent.parent / "data" / "inquiries"


def read(name: str) -> str:
    return (INQUIRIES / f"{name}.txt").read_text()


# ---- extraction (mock mode runs the real Pydantic AI agent + validators) ----

def test_complete_inquiry_extracts_every_field():
    lead = extract_lead(read("complete"), "mock")
    assert lead.customer_name == "Denise Carter"
    assert lead.phone == "(205) 555-0147"
    assert lead.email == "denise.carter@example.com"
    assert lead.service_address == "1412 Oak Hollow Lane, Hoover, AL 35226"
    assert lead.service_type == "repair"
    assert lead.system_type == "central_ac"
    assert lead.callback_preference == (
        "Best time to reach me is after 5:30 pm on weekdays, please text first before calling."
    )


@pytest.mark.parametrize("name", ["complete", "missing_phone", "ambiguous_service", "declined_follow_up"])
def test_callback_preference_is_word_for_word(name):
    inquiry = read(name)
    lead = extract_lead(inquiry, "mock")
    assert lead.callback_preference is not None
    assert lead.callback_preference in inquiry


def test_missing_phone_stays_empty():
    lead = extract_lead(read("missing_phone"), "mock")
    assert lead.phone is None
    assert lead.email == "mbell.home@example.net"


def test_ambiguous_service_is_not_guessed():
    lead = extract_lead(read("ambiguous_service"), "mock")
    assert lead.service_type is None
    assert set(lead.service_type_candidates) == {"repair", "replacement"}
    assert lead.service_address is None
    assert lead.email is None
    assert lead.system_type is None


def test_declined_follow_up_is_detected():
    lead = extract_lead(read("declined_follow_up"), "mock")
    assert lead.declined_follow_up is True
    assert "please don't call or email me" in lead.callback_preference


def test_empty_inquiry_is_rejected():
    with pytest.raises(ValueError, match="enter a customer inquiry"):
        extract_lead("   ", "mock")


# ---- grounding: a paraphrased callback preference is sent back to the model ----

def scripted_model(*responses: dict) -> tuple[FunctionModel, list]:
    calls = []

    def respond(messages, info):
        calls.append(messages)
        args = responses[min(len(calls), len(responses)) - 1]
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=args)])

    return FunctionModel(respond), calls


def test_paraphrased_callback_triggers_retry_then_passes():
    inquiry = read("complete")
    good = rule_based_extract(inquiry)
    paraphrased = {**good, "callback_preference": "Text after 5:30pm on weekdays."}
    model, calls = scripted_model(paraphrased, good)

    agent = build_agent("mock")
    with agent.override(model=model):
        lead = agent.run_sync(inquiry, deps=inquiry).output

    assert len(calls) == 2
    assert lead.callback_preference == good["callback_preference"]


def test_invented_phone_is_never_accepted():
    inquiry = read("missing_phone")
    invented = {**rule_based_extract(inquiry), "phone": "205-555-9999"}
    model, _ = scripted_model(invented)

    agent = build_agent("mock")
    with agent.override(model=model), pytest.raises(UnexpectedModelBehavior):
        agent.run_sync(inquiry, deps=inquiry)


# ---- schema rules ----

def test_schema_rejects_bad_phone():
    with pytest.raises(ValidationError, match="10 digits"):
        HVACLead(intent="hvac_service", phone="555-0147")


def test_schema_requires_words_when_declined():
    with pytest.raises(ValidationError, match="customer's words"):
        HVACLead(intent="hvac_service", declined_follow_up=True)


def test_blank_strings_become_unknown():
    lead = HVACLead(intent="hvac_service", email="  ", customer_name="")
    assert lead.email is None and lead.customer_name is None


def test_grounding_flags_unsupported_name():
    lead = HVACLead(intent="hvac_service", customer_name="Robert")
    assert grounding_problems(lead, "Hi, my AC is broken.")


# ---- drafting and outbound ----

def test_declined_customer_gets_no_draft():
    review = process_inquiry(read("declined_follow_up"), "mock")
    assert review.follow_up.status == "not_drafted"
    assert review.follow_up.body is None
    assert "customer_declined_follow_up" in review.follow_up.review_flags


def test_every_draft_waits_for_contractor_and_is_unsent():
    for path in INQUIRIES.glob("*.txt"):
        draft = process_inquiry(path.read_text(), "mock").follow_up
        assert draft.outbound_sent is False
        assert draft.status in ("pending_contractor_review", "not_drafted")


def test_outbound_is_hard_disabled():
    assert outbound.OUTBOUND_ENABLED is False
    draft = process_inquiry(read("complete"), "mock").follow_up
    with pytest.raises(outbound.OutboundDisabledError):
        outbound.send_follow_up(draft)


class FakeAnthropic:
    """Simulate the Claude Messages API, like FleetMind's MockClient."""

    def __init__(self, text):
        self.text = text
        self.kwargs = None
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def test_live_draft_uses_claude_and_stays_in_review():
    lead = extract_lead(read("missing_phone"), "mock")
    fake = FakeAnthropic("Hi Marcus, this is [Your Name] with [Company Name]...")
    draft = drafting.draft_follow_up(lead, "live", client=fake)

    assert fake.kwargs["model"] == config.live_model_id()
    assert "mbell.home@example.net" in fake.kwargs["messages"][0]["content"]
    assert draft.status == "pending_contractor_review"
    assert draft.channel == "email"
    assert draft.outbound_sent is False


def test_live_draft_flags_prices():
    lead = extract_lead(read("complete"), "mock")
    draft = drafting.draft_follow_up(lead, "live", client=FakeAnthropic("A visit is $89."))
    assert "draft_mentions_price" in draft.review_flags


def test_live_mode_needs_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is missing"):
        config.get_mode("live")


def test_mock_is_the_default(monkeypatch):
    monkeypatch.delenv("LEAD_DEMO_MODE", raising=False)
    assert config.get_mode() == "mock"
