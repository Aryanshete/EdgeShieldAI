# Architecture

The current implemented flow is:

`video → YOLOv8 detection → ByteTrack person tracking → zone analysis → events → context → risk → reasoning → incident → dashboard`

## Tracking boundary

`src/tracker.py` uses ByteTrack to retain an anonymous ID across consecutive
frames from one camera view. It maintains the track history required by later
stages:

```text
track_id → first_seen, last_seen, positions, zones, events
```

Hard edits or camera changes produce a new ID because identity continuity has
not been visually established. EdgeShield does not merge those IDs or claim a
person is the same individual across a cut. A continuous fixed-camera video is
therefore required for the restricted-zone MVP.
