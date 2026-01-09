"""demo_envelope.py

Author: Samir "Sam" Gabsi
License: MIT

Run:
    python demo_envelope.py

This demo shows:
- Envelope creation
- Attaching a governance decision
- Setting temporal constraints
- Executing and emitting output
- Printing JSON and a compact audit record
"""

from datetime import timedelta
from hil_aiw_envelope.envelope import Envelope, Governance, isoformat_z, utc_now


def main() -> None:
    env = Envelope.create(
        actor_type="human",
        actor_id="DEMO_USER",
        intent_type="action",
        intent_label="plan_next_steps",
        input_data={"topic": "public shareable envelope contract"},
        routing_target="agent:planner",
    )

    env.governance = Governance(
        policy_version="policy-2026.01",
        decision="allowed",
        reason="Low risk, safe operation",
        risk_score=0.08,
        triggered_rules=["SAFE_DEFAULTS"],
    )
    env.temporal.mark_accepted()

    now = utc_now()
    env.temporal.due_by = isoformat_z(now + timedelta(seconds=30))
    env.temporal.hard_deadline = isoformat_z(now + timedelta(seconds=60))
    env.temporal.maximum_wallclock_seconds = 10

    env.state.status = "executing"
    env.temporal.mark_started()

    env.set_output({"plan": ["Publish envelope.py", "Include demo", "Add README"], "ok": True})
    env.validate()

    print("=== Envelope JSON ===")
    print(env.to_json(indent=2))

    print("\n=== Audit Record (compact) ===")
    print(env.to_audit_record())

    print("\n=== Deadline status ===")
    status, msg = env.temporal.deadline_status()
    print(status, msg or "")


if __name__ == "__main__":
    main()
