"""Run mode and model settings.

Mock mode is the default so the demo never makes a paid call by accident.
"""

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

Mode = Literal["mock", "live"]
DEFAULT_MODEL_ID = "claude-sonnet-5-5"


def get_mode(override: str | None = None) -> Mode:
    mode = (override or os.getenv("LEAD_DEMO_MODE") or "mock").strip().lower()
    if mode not in ("mock", "live"):
        raise ValueError("LEAD_DEMO_MODE must be 'mock' or 'live'.")
    if mode == "live" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Add it to your .env file, "
            "or run in mock mode."
        )
    return mode  # type: ignore[return-value]


def live_model_id() -> str:
    return os.getenv("ANTHROPIC_MODEL_ID", DEFAULT_MODEL_ID)
