"""Inquiry -> validated lead -> follow-up draft for contractor review."""

from .config import get_mode
from .drafting import draft_follow_up
from .extraction import extract_lead
from .schemas import LeadReview


def process_inquiry(inquiry: str, mode: str | None = None) -> LeadReview:
    run_mode = get_mode(mode)
    lead = extract_lead(inquiry, run_mode)
    follow_up = draft_follow_up(lead, run_mode)
    return LeadReview(mode=run_mode, inquiry=inquiry.strip(), lead=lead, follow_up=follow_up)
