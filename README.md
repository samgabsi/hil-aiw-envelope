# HIL-AIW Envelope (Public, Shareable)

**Author:** Samir "Sam" Gabsi  
**License:** MIT (see `LICENSE`)

This repo is a **clean, stdlib-only** reference implementation of a message **Envelope** pattern.

It is designed to be publicly shareable: it contains **no governance rubrics**, no orchestration logic,
no internal keys, and no proprietary code paths. It's a portable contract you can embed inside any
orchestrator / agent runtime / workflow engine.

## What is an Envelope?

An *Envelope* is a single, auditable object that wraps a unit of work (or observation) with:

- **Actor**: who initiated it
- **Intent**: what it is trying to do
- **Routing**: where it should go
- **Governance**: policy decision & risk summary
- **Temporal**: timestamps + deadlines + SLAs
- **State**: lifecycle status & errors
- **Cost**: optional usage/cost tracking
- **Payload**: `input` and `output` dictionaries
- **meta**: extension point for future fields

## Files

- `hil_aiw_envelope/envelope.py` — the envelope implementation
- `demo_envelope.py` — a single runnable demo module
- `LICENSE` — MIT license text

## Quickstart

Requirements: Python 3.10+ (works on 3.9+ with minor typing adjustments)

```bash
python demo_envelope.py
```

You should see:
- a full JSON envelope
- a compact `audit_record` projection

## API Highlights

**Create:**
```python
env = Envelope.create(
    actor_type="human",
    actor_id="DEMO_USER",
    intent_type="action",
    intent_label="plan_next_steps",
    input_data={"hello": "world"},
    routing_target="agent:planner",
)
```

**Serialize / Deserialize:**
```python
json_str = env.to_json()
env2 = Envelope.from_json(json_str)
```

**Audit projection:**
```python
record = env.to_audit_record()
```

**Temporal constraints:**
- `earliest_start`: don't begin before this time
- `due_by`: soft deadline (warn / soft miss)
- `hard_deadline`: escalation deadline (hard miss)
- `maximum_wallclock_seconds`: execution time cap

## Safety

This module is intentionally **not** an orchestrator and **does not** execute tools.
It is a contract layer only.
