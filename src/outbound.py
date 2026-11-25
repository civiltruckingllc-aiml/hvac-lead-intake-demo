"""Outbound messaging is switched off in this demo.

This is a hard-coded constant on purpose: no environment variable or flag
can turn sending on. Drafts only go to the contractor for review.
"""

from .schemas import FollowUpDraft

OUTBOUND_ENABLED = False


class OutboundDisabledError(RuntimeError):
    """Raised whenever anything tries to send a message to a customer."""


def send_follow_up(draft: FollowUpDraft) -> None:
    raise OutboundDisabledError(
        "Outbound messaging is disabled. The contractor must review and "
        "send follow-ups manually."
    )
