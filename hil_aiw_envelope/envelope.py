"""hil_aiw_envelope.envelope

Author
------
Samir "Sam" Gabsi

Overview
--------
A clean, dependency-free (stdlib-only) **Envelope** implementation designed to be
publicly shareable while still being genuinely useful.

Why an Envelope?
---------------
In agentic / workflow systems, you want a single, auditable unit that carries:

- **Who** initiated something (Actor)
- **What** they intended (Intent)
- **Where** it should go (Routing)
- **What** policy decided about it (Governance)
- **When** it must run / expire (Temporal)
- **How** it progressed (State)
- **How much** it cost (Cost)
- The **payload** (input/output) and metadata

This module provides a small, professional contract you can embed in:
queues, HTTP APIs, event buses, orchestrators, audit pipelines, and storage layers.

Design goals
------------
- **Shareable**: no secrets, no internal business logic, no vendor lock-in
- **Usable**: typed dataclasses, clear docs, JSON-friendly serialization
- **Auditable**: stable IDs, timestamps, and an explicit audit record projection
- **Extensible**: `meta` dict and versioning to support evolution

License
-------
MIT (see LICENSE)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
import json
import uuid

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

ENVELOPE_SCHEMA_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    """Return a random UUID4 string."""
    return str(uuid.uuid4())


def isoformat_z(dt: datetime) -> str:
    """Format an aware datetime as an ISO-8601 string with a UTC 'Z' suffix."""
    if dt.tzinfo is None:
        raise ValueError("isoformat_z requires an aware datetime")
    dt_utc = dt.astimezone(timezone.utc)
    s = dt_utc.isoformat(timespec="milliseconds")
    return s.replace("+00:00", "Z")


def parse_iso8601(s: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp is naive (missing timezone): {s}")
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Actor:
    """Who initiated or is responsible for this envelope."""
    type: str
    id: str
    display_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "id": self.id, "display_name": self.display_name}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Actor":
        return cls(
            type=str(d.get("type", "system")),
            id=str(d.get("id", "unknown")),
            display_name=d.get("display_name"),
        )


@dataclass(frozen=True)
class Intent:
    """What the envelope is trying to do."""
    type: str
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "label": self.label}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Intent":
        return cls(type=str(d.get("type", "prompt")), label=str(d.get("label", "")))


@dataclass
class Routing:
    """Where the envelope should go."""
    target: Optional[str] = None
    brain_id: Optional[str] = None
    atlas_id: Optional[str] = None
    strategy: Optional[str] = None
    hop_count: int = 0
    max_hops: int = 24
    upstream_envelope_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "brain_id": self.brain_id,
            "atlas_id": self.atlas_id,
            "strategy": self.strategy,
            "hop_count": self.hop_count,
            "max_hops": self.max_hops,
            "upstream_envelope_id": self.upstream_envelope_id,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Routing":
        return cls(
            target=d.get("target"),
            brain_id=d.get("brain_id"),
            atlas_id=d.get("atlas_id"),
            strategy=d.get("strategy"),
            hop_count=int(d.get("hop_count", 0)),
            max_hops=int(d.get("max_hops", 24)),
            upstream_envelope_id=d.get("upstream_envelope_id"),
        )


@dataclass
class Governance:
    """Governance decision and rationale."""
    policy_version: str
    decision: str
    reason: Optional[str] = None
    risk_score: float = 0.0
    triggered_rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "decision": self.decision,
            "reason": self.reason,
            "risk_score": self.risk_score,
            "triggered_rules": list(self.triggered_rules),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Governance":
        return cls(
            policy_version=str(d.get("policy_version", "unknown")),
            decision=str(d.get("decision", "allowed")),
            reason=d.get("reason"),
            risk_score=float(d.get("risk_score", 0.0)),
            triggered_rules=list(d.get("triggered_rules", [])),
        )


@dataclass
class Temporal:
    """Time semantics and constraints."""
    created_at: str = field(default_factory=lambda: isoformat_z(utc_now()))
    accepted_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    earliest_start: Optional[str] = None
    due_by: Optional[str] = None
    hard_deadline: Optional[str] = None
    maximum_wallclock_seconds: Optional[int] = None
    review_by: Optional[str] = None
    retention_until: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "accepted_at": self.accepted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "earliest_start": self.earliest_start,
            "due_by": self.due_by,
            "hard_deadline": self.hard_deadline,
            "maximum_wallclock_seconds": self.maximum_wallclock_seconds,
            "review_by": self.review_by,
            "retention_until": self.retention_until,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Temporal":
        t = cls(created_at=str(d.get("created_at", isoformat_z(utc_now()))))
        t.accepted_at = d.get("accepted_at")
        t.started_at = d.get("started_at")
        t.completed_at = d.get("completed_at")
        t.earliest_start = d.get("earliest_start")
        t.due_by = d.get("due_by")
        t.hard_deadline = d.get("hard_deadline")
        t.maximum_wallclock_seconds = d.get("maximum_wallclock_seconds")
        t.review_by = d.get("review_by")
        t.retention_until = d.get("retention_until")
        return t

    def mark_accepted(self) -> None:
        self.accepted_at = isoformat_z(utc_now())

    def mark_started(self) -> None:
        self.started_at = isoformat_z(utc_now())

    def mark_completed(self) -> None:
        self.completed_at = isoformat_z(utc_now())

    def elapsed_seconds(self, now: Optional[datetime] = None) -> Optional[float]:
        if self.started_at is None:
            return None
        now_dt = (now or utc_now()).astimezone(timezone.utc)
        started = parse_iso8601(self.started_at)
        return (now_dt - started).total_seconds()

    def wallclock_exceeded(self, now: Optional[datetime] = None) -> bool:
        if self.maximum_wallclock_seconds is None:
            return False
        elapsed = self.elapsed_seconds(now=now)
        return (elapsed is not None) and (elapsed > float(self.maximum_wallclock_seconds))

    def deadline_status(self, now: Optional[datetime] = None, warn_at: float = 0.2) -> Tuple[str, Optional[str]]:
        now_dt = (now or utc_now()).astimezone(timezone.utc)

        if self.hard_deadline:
            hard = parse_iso8601(self.hard_deadline)
            if now_dt > hard:
                return "hard_missed", f"Hard deadline passed at {self.hard_deadline}"

        if self.due_by:
            due = parse_iso8601(self.due_by)
            if now_dt > due:
                return "soft_missed", f"Soft deadline passed at {self.due_by}"
            try:
                created = parse_iso8601(self.created_at)
                total = (due - created).total_seconds()
                remaining = (due - now_dt).total_seconds()
                if total > 0 and (remaining / total) < warn_at:
                    return "warning", f"Approaching due_by: ~{int(remaining)}s remaining"
            except Exception:
                pass

        return "ok", None


@dataclass
class Cost:
    """Optional cost/usage tracking."""
    estimated_usd: float = 0.0
    actual_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_usd": self.estimated_usd,
            "actual_usd": self.actual_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Cost":
        return cls(
            estimated_usd=float(d.get("estimated_usd", 0.0)),
            actual_usd=float(d.get("actual_usd", 0.0)),
            input_tokens=int(d.get("input_tokens", 0)),
            output_tokens=int(d.get("output_tokens", 0)),
            model=d.get("model"),
        )


@dataclass
class State:
    """Lifecycle state for processors/orchestrators."""
    status: str = "created"
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "error": self.error, "error_type": self.error_type}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "State":
        return cls(
            status=str(d.get("status", "created")),
            error=d.get("error"),
            error_type=d.get("error_type"),
        )


@dataclass
class Envelope:
    """The canonical message object."""
    envelope_id: str
    schema_version: str
    session_id: str
    timestamp: str

    actor: Actor
    intent: Intent
    routing: Routing = field(default_factory=Routing)
    governance: Optional[Governance] = None
    temporal: Temporal = field(default_factory=Temporal)
    state: State = field(default_factory=State)
    cost: Cost = field(default_factory=Cost)

    input: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        actor_type: str,
        actor_id: str,
        intent_type: str,
        intent_label: str,
        session_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        routing_target: Optional[str] = None,
    ) -> "Envelope":
        sid = session_id or uuid.uuid4().hex[:10]
        env = cls(
            envelope_id=new_uuid(),
            schema_version=ENVELOPE_SCHEMA_VERSION,
            session_id=sid,
            timestamp=isoformat_z(utc_now()),
            actor=Actor(type=actor_type, id=actor_id),
            intent=Intent(type=intent_type, label=intent_label),
            input=input_data or {},
        )
        if routing_target:
            env.routing.target = routing_target
        return env

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "envelope_id": self.envelope_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "actor": self.actor.to_dict(),
            "intent": self.intent.to_dict(),
            "routing": self.routing.to_dict(),
            "governance": self.governance.to_dict() if self.governance else None,
            "temporal": self.temporal.to_dict(),
            "state": self.state.to_dict(),
            "cost": self.cost.to_dict(),
            "input": self.input,
            "output": self.output,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Envelope":
        return cls(
            envelope_id=str(d.get("envelope_id", new_uuid())),
            schema_version=str(d.get("schema_version", ENVELOPE_SCHEMA_VERSION)),
            session_id=str(d.get("session_id", uuid.uuid4().hex[:10])),
            timestamp=str(d.get("timestamp", isoformat_z(utc_now()))),
            actor=Actor.from_dict(d.get("actor", {})),
            intent=Intent.from_dict(d.get("intent", {})),
            routing=Routing.from_dict(d.get("routing", {})),
            governance=Governance.from_dict(d["governance"]) if d.get("governance") else None,
            temporal=Temporal.from_dict(d.get("temporal", {})),
            state=State.from_dict(d.get("state", {})),
            cost=Cost.from_dict(d.get("cost", {})),
            input=dict(d.get("input", {}) or {}),
            output=dict(d.get("output", {}) or {}),
            meta=dict(d.get("meta", {}) or {}),
        )

    def to_json(self, *, indent: int = 2, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_json(cls, s: str) -> "Envelope":
        return cls.from_dict(json.loads(s))

    def set_output(self, output: Dict[str, Any]) -> None:
        self.output = output
        self.state.status = "executed"
        self.temporal.mark_completed()

    def set_error(self, message: str, *, error_type: str = "Error") -> None:
        self.state.status = "errored"
        self.state.error = message
        self.state.error_type = error_type
        self.temporal.mark_completed()

    def bump_hop(self) -> None:
        self.routing.hop_count += 1
        if self.routing.hop_count > self.routing.max_hops:
            raise RuntimeError(f"Max hops exceeded ({self.routing.max_hops}) for envelope {self.envelope_id}")

    def validate(self) -> None:
        if not self.envelope_id:
            raise ValueError("envelope_id is required")
        if not self.session_id:
            raise ValueError("session_id is required")
        if not self.actor.type or not self.actor.id:
            raise ValueError("actor.type and actor.id are required")
        if not self.intent.type:
            raise ValueError("intent.type is required")
        _ = parse_iso8601(self.timestamp)
        _ = parse_iso8601(self.temporal.created_at)

    def to_audit_record(self) -> Dict[str, Any]:
        decision = self.governance.decision if self.governance else "none"
        risk = self.governance.risk_score if self.governance else 0.0
        total_tokens = int(self.cost.input_tokens) + int(self.cost.output_tokens)
        return {
            "schema_version": self.schema_version,
            "envelope_id": self.envelope_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "actor": f"{self.actor.type}:{self.actor.id}",
            "intent": f"{self.intent.type}:{self.intent.label}",
            "target": self.routing.target,
            "governance_decision": decision,
            "governance_risk": risk,
            "status": self.state.status,
            "error": self.state.error,
            "cost_usd": self.cost.actual_usd,
            "model": self.cost.model,
            "tokens": total_tokens,
        }


__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "Actor",
    "Intent",
    "Routing",
    "Governance",
    "Temporal",
    "Cost",
    "State",
    "Envelope",
    "utc_now",
    "isoformat_z",
    "parse_iso8601",
]
