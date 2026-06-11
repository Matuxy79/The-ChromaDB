# CLS Safety Flags

This prototype is not a safety authority. It is a local evidence assistant that can help locate relevant text in indexed CLS documents.

## What The Safety Layer Does

`examples/cls_safety.py` checks each user query for conservative keyword triggers such as:

- beryllium
- radiation and interlocks
- lasers
- cryostat / cryogenics
- high voltage
- vacuum failure
- fire, medical, evacuation, injury

When a trigger matches, the UI keeps the answer grounded in retrieved evidence and should point the user to beamline staff or emergency contacts when evidence is incomplete. Optional carrier cleanup is still limited to mechanical text cleanup; it is not allowed to improvise procedures.

The interface should still be read conservatively:

- answer only from retrieved evidence
- do not infer or improvise procedures from partial snippets
- contact beamline staff if the evidence is incomplete
- follow emergency contacts first for active emergencies

## What It Does Not Do

- It does not certify that an answer is safe.
- It does not replace beamline staff, training, permits, lockouts, posted procedures, or control-room instructions.
- It does not prove a document is current.

## Emergency Contacts

The current prototype includes contacts extracted from the IVU beamline manual. Treat these as configuration data that must be reviewed before production use.

| Contact | Number | Note |
| ------- | ------ | ---- |
| Fire / Ambulance | 911 | Serious emergency |
| U-Sask Security | 9-306-966-5555 | From a CLS phone |
| CLS Control Room | ext. 3570 | Operations / beam |
| Floor Coordinator | ext. 3639 | Hutch access / floor |
| Health & Safety (HSE) | ext. 3663 | Reportable incidents |
| Beamline (SOE-3 / IVU) | ext. 3832 | In-hutch phone |

## Low-Confidence Flag

The retrieval layer marks an answer low-confidence when the best Chroma distance is above `0.55`. This threshold lives in `examples/cls_safety.py` as `LOW_CONFIDENCE_DISTANCE`.

The user should then open the retrieval trace and inspect the source preview before relying on the answer.

## Recommended Production Safeties

- Keep a human-reviewed safety-contact file outside code.
- Add document version/date metadata and display it in the retrieval trace.
- Add a mandatory "call staff" response for active emergency language.
- Add tests for every safety keyword category.
- Never show safety-critical guidance without retrieved evidence and visible source trace.
