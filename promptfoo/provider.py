"""Promptfoo provider: runs the full lead pipeline on the rendered inquiry.

Mode comes from LEAD_DEMO_MODE (default "mock", so evals cost nothing).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import process_inquiry  # noqa: E402


def call_api(prompt, options, context):
    try:
        review = process_inquiry(prompt, os.getenv("LEAD_DEMO_MODE"))
    except Exception as exc:  # report to Promptfoo as a failed case
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"output": review.model_dump_json()}
