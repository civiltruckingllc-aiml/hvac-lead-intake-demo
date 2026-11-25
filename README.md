# HVAC Lead-Intake Demo

Takes a homeowner's HVAC inquiry, pulls out checked lead fields with **Pydantic AI**,
and uses the **Claude API** to draft a follow-up message for the contractor to review.
**Nothing is ever sent.** Sending is switched off in the code (`src/outbound.py`).

The style follows FleetMind AI's `src/llm.py`: one Pydantic model, `None` for anything
the customer didn't say, a "never invent details" rule, plain `ValueError` messages,
and tests that use mock clients.

## Run it

```bash
pip install -r requirements.txt
python demo.py                 # all 4 sample inquiries, mock mode (free)
python demo.py --json data/inquiries/complete.txt
```

Live mode (paid API calls): copy `.env.example` to `.env`, add your key, then
`python demo.py --live` or set `LEAD_DEMO_MODE=live`.

## How it works

1. **Extract** (`src/extraction.py`): a Pydantic AI agent returns an `HVACLead`.
   A second check compares the result to the inquiry text. If the model rewords the
   callback preference or problem description, or writes down a name, phone number,
   or email the customer never gave, the check sends the output back to the model to fix.
   If the model still gets it wrong after 2 retries, the run fails.
2. **Draft** (`src/drafting.py`): Claude writes a text, an email, or call talking
   points, depending on what the customer asked for. If the customer said not to contact
   them, or gave no phone or email, no draft is written.
3. **Review**: every result is either `pending_contractor_review` or `not_drafted`,
   with flags for the contractor such as `missing_phone` or `ambiguous_service_type`.

**Mock mode** swaps the paid models for simple rules. It still runs the real Pydantic AI
agent, the real schema, and the real wording checks.

## Tests

```bash
python -m pytest -v            # 22 offline tests
npx promptfoo eval             # 4 Promptfoo cases, mock mode
LEAD_DEMO_MODE=live npx promptfoo eval   # same cases against the real models
```

| Promptfoo case | What it checks |
|---|---|
| Complete inquiry | All fields filled in; callback words match exactly; text draft ready for review |
| Missing phone number | `phone` is empty; reply goes by email; `missing_phone` flag |
| Ambiguous service request | No service type guessed; both options listed; draft asks which one |
| Customer declines follow-up | Decline detected; their words kept; no draft written |

Every case also checks that nothing was sent and that the callback words appear exactly
as they do in the inquiry.

## Files

```
demo.py                 command-line demo
src/schemas.py          HVACLead, FollowUpDraft, LeadReview, wording checks
src/extraction.py       Pydantic AI agent and instructions
src/drafting.py         Claude API follow-up draft and review flags
src/outbound.py         sending switched off (hard-coded)
src/mock_models.py      offline rule-based model for mock mode
src/config.py           mock or live mode, model ID
data/inquiries/         4 made-up example inquiries
promptfooconfig.yaml    Promptfoo cases; promptfoo/provider.py runs the pipeline
tests/                  pytest suite
```
