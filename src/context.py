"""Temporal event analysis and contextual sequence modeling for EdgeShield."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.events import (
    EVENT_ABANDONED_OBJECT,
    EVENT_AFTER_HOURS_ACTIVITY,
    EVENT_LOITERING_DETECTED,
    EVENT_OBJECT_INTERACTION,
    EVENT_PERSON_DETECTED,
    EVENT_RESTRICTED_ZONE_DWELL,
    EVENT_RESTRICTED_ZONE_ENTRY,
    EVENT_RESTRICTED_ZONE_EXIT,
    EVENT_UNUSUAL_MOVEMENT,
    EVENT_ZONE_APPROACH,
    SecurityEvent,
)
from src.zones import Zone, ZoneManager, is_restricted_zone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTEXT_FILE = PROJECT_ROOT / "data" / "context.json"

# Canonical token mapping for downstream risk and LLM consumption
CANONICAL_EVENT_TOKENS: dict[str, str] = {
    EVENT_PERSON_DETECTED: "person_detected",
    EVENT_ZONE_APPROACH: "zone_approach",
    EVENT_RESTRICTED_ZONE_ENTRY: "restricted_zone_entry",
    EVENT_AFTER_HOURS_ACTIVITY: "after_hours_activity",
    EVENT_RESTRICTED_ZONE_DWELL: "extended_dwell",
    EVENT_OBJECT_INTERACTION: "object_interaction",
    EVENT_RESTRICTED_ZONE_EXIT: "restricted_zone_exit",
    EVENT_LOITERING_DETECTED: "loitering_detected",
    EVENT_ABANDONED_OBJECT: "abandoned_object_detected",
    EVENT_UNUSUAL_MOVEMENT: "unusual_movement",
}


@dataclass(frozen=True)
class SecurityContext:
    """Aggregated temporal context representing a coherent sequence of security events."""

    location: str
    time: str
    authorized_hours: str
    track_id: int
    events: list[str]
    event_descriptions: list[str]
    start_timestamp: str
    end_timestamp: str
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return structured context matching the specification schema."""
        return {
            "location": self.location,
            "time": self.time,
            "authorized_hours": self.authorized_hours,
            "track_id": self.track_id,
            "events": list(self.events),
            "event_descriptions": list(self.event_descriptions),
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": round(self.duration_seconds, 2),
            "metadata": dict(self.metadata),
        }


def format_event_description(event: SecurityEvent) -> str:
    """Format a factual description of a security event grounded strictly in visual evidence."""
    tid = event.track_id
    etype = event.event_type
    zone_name = event.zone or "monitored zone"

    if etype == EVENT_PERSON_DETECTED:
        return f"Person #{tid:02d} detected in scene"
    elif etype == EVENT_ZONE_APPROACH:
        dist = event.metadata.get("distance_pixels")
        dist_str = f" ({dist}px)" if dist else ""
        return f"Person #{tid:02d} approached restricted zone '{zone_name}'{dist_str}"
    elif etype == EVENT_RESTRICTED_ZONE_ENTRY:
        return f"Person #{tid:02d} entered restricted zone"
    elif etype == EVENT_AFTER_HOURS_ACTIVITY:
        return "Activity occurred outside authorized hours"
    elif etype == EVENT_RESTRICTED_ZONE_DWELL:
        dwell = event.metadata.get("dwell_seconds", 0)
        return f"Person remained inside for {int(dwell)} seconds"
    elif etype == EVENT_OBJECT_INTERACTION:
        return "Person approached an unattended object"
    elif etype == EVENT_RESTRICTED_ZONE_EXIT:
        total = event.metadata.get("total_dwell_seconds")
        if total:
            return f"Person exited restricted zone after {int(total)} seconds"
        return "Person exited restricted zone"
    elif etype == EVENT_LOITERING_DETECTED:
        dur = event.metadata.get("loitering_duration", 0)
        return f"Person #{tid:02d} lingered in {zone_name} for {int(dur)} seconds"
    elif etype == EVENT_ABANDONED_OBJECT:
        cname = event.metadata.get("class_name", "object")
        return f"Unattended stationary {cname} detected in {zone_name}"
    elif etype == EVENT_UNUSUAL_MOVEMENT:
        utype = event.metadata.get("type", "unusual movement")
        return f"Person #{tid:02d} exhibited {utype.replace('_', ' ')} in {zone_name}"
    return f"Person #{tid:02d} triggered {etype} in {zone_name}"


class ContextEngine:
    """Synthesizes chronological security events into coherent multi-event contexts."""

    def __init__(self, zone_manager: ZoneManager | None = None) -> None:
        self.zone_manager = zone_manager or ZoneManager()

    def _resolve_authorized_hours(self, zone_name: str | None) -> str:
        """Look up operating window string for a zone name."""
        if zone_name:
            for zone in self.zone_manager.zones:
                if zone.name == zone_name and zone.authorized_hours:
                    start = zone.authorized_hours.get("start", "08:00")
                    end = zone.authorized_hours.get("end", "18:00")
                    return f"{start}-{end}"
        return "08:00-18:00"

    def build_context_for_track(
        self,
        track_id: int,
        events: Sequence[SecurityEvent],
    ) -> SecurityContext | None:
        """Build a SecurityContext from an ordered sequence of events for a single track ID."""
        track_events = [e for e in events if e.track_id == track_id]
        if not track_events:
            return None

        # Sort events by timestamp_seconds to ensure strict chronological order
        sorted_events = sorted(track_events, key=lambda e: e.timestamp_seconds)

        # 1. Determine primary location
        # Priority: Restricted zone mentioned in events > any named zone > default
        primary_location = "Monitored Area"
        for ev in sorted_events:
            if ev.zone:
                primary_location = ev.zone
                zone_obj = self.zone_manager.get_zone_by_id(ev.metadata.get("zone_id", ""))
                if zone_obj and is_restricted_zone(zone_obj):
                    primary_location = zone_obj.name
                    break

        # 2. Derive time (HH:MM)
        start_ts = sorted_events[0].timestamp
        time_hh_mm = ":".join(start_ts.split(":")[:2]) if ":" in start_ts else start_ts
        end_ts = sorted_events[-1].timestamp
        duration = max(0.0, sorted_events[-1].timestamp_seconds - sorted_events[0].timestamp_seconds)

        # 3. Derive authorized hours for the location
        authorized_hours = self._resolve_authorized_hours(primary_location)

        # 4. Canonical event tokens (deduplicating adjacent identical tokens)
        canonical_tokens: list[str] = []
        for ev in sorted_events:
            token = CANONICAL_EVENT_TOKENS.get(ev.event_type, ev.event_type)
            if not canonical_tokens or canonical_tokens[-1] != token:
                canonical_tokens.append(token)

        # 5. Narrative descriptions
        descriptions: list[str] = []
        for ev in sorted_events:
            desc = format_event_description(ev)
            if not descriptions or descriptions[-1] != desc:
                descriptions.append(desc)

        return SecurityContext(
            location=primary_location,
            time=time_hh_mm,
            authorized_hours=authorized_hours,
            track_id=track_id,
            events=canonical_tokens,
            event_descriptions=descriptions,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            duration_seconds=duration,
            metadata={
                "events_count": len(sorted_events),
                "zones_involved": list({e.zone for e in sorted_events if e.zone}),
            },
        )

    def build_all_contexts(
        self,
        events: Sequence[SecurityEvent],
    ) -> list[SecurityContext]:
        """Group events by track ID and construct a SecurityContext for each observed subject."""
        unique_track_ids: list[int] = []
        for e in events:
            if e.track_id not in unique_track_ids:
                unique_track_ids.append(e.track_id)

        contexts: list[SecurityContext] = []
        for tid in unique_track_ids:
            ctx = self.build_context_for_track(tid, events)
            if ctx is not None:
                contexts.append(ctx)
        return contexts

    def export_contexts(
        self,
        contexts: Sequence[SecurityContext],
        output_path: str | Path = DEFAULT_CONTEXT_FILE,
    ) -> None:
        """Export serialized contexts to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [ctx.as_dict() for ctx in contexts]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    @classmethod
    def load_contexts(cls, path: str | Path = DEFAULT_CONTEXT_FILE) -> list[SecurityContext]:
        """Load contexts from a JSON file into SecurityContext objects."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Context file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        contexts: list[SecurityContext] = []
        for item in data:
            contexts.append(
                SecurityContext(
                    location=item["location"],
                    time=item["time"],
                    authorized_hours=item["authorized_hours"],
                    track_id=item["track_id"],
                    events=list(item.get("events", [])),
                    event_descriptions=list(item.get("event_descriptions", [])),
                    start_timestamp=item.get("start_timestamp", ""),
                    end_timestamp=item.get("end_timestamp", ""),
                    duration_seconds=float(item.get("duration_seconds", 0.0)),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return contexts
