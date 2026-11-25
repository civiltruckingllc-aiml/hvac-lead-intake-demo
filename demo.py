"""Run the HVAC lead-intake demo on one or more inquiry files.

    python demo.py                         # all sample inquiries, mock mode
    python demo.py data/inquiries/complete.txt
    python demo.py --live data/inquiries/complete.txt   # paid Claude API calls
    python demo.py --json ...              # raw JSON instead of the review card
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import process_inquiry
from src.schemas import LeadReview

SAMPLES = sorted(Path("data/inquiries").glob("*.txt"))


def show(review: LeadReview, name: str) -> None:
    lead, draft = review.lead, review.follow_up
    print("=" * 70)
    print(f"{name}   [mode: {review.mode}]")
    print("-" * 70)
    print("INQUIRY:", review.inquiry)
    print("\nLEAD FIELDS (blank = customer didn't say):")
    for field, value in lead.model_dump().items():
        shown = "" if value in (None, []) else value
        print(f"  {field:<24} {shown}")
    print(f"\nFOLLOW-UP: {draft.status}   (outbound_sent={draft.outbound_sent})")
    if draft.review_flags:
        print("  review flags:", ", ".join(draft.review_flags))
    if draft.reason_not_drafted:
        print("  reason:", draft.reason_not_drafted)
    if draft.body:
        print(f"  channel: {draft.channel}" + (f"   subject: {draft.subject}" if draft.subject else ""))
        print("  " + draft.body.replace("\n", "\n  "))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, default=SAMPLES)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", help="no paid calls (default)")
    group.add_argument("--live", action="store_true", help="use Pydantic AI + Claude API")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    mode = "live" if args.live else "mock" if args.mock else None
    for path in args.files:
        review = process_inquiry(path.read_text(), mode)
        print(review.model_dump_json(indent=2)) if args.json else show(review, path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
