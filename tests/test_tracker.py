"""Unit tests for Phase 2 tracking utilities without model inference."""

from __future__ import annotations

import numpy as np

from src.tracker import ObjectTracker, bottom_center, update_track_history


def test_bottom_center_uses_the_bottom_of_the_box() -> None:
    assert bottom_center([10, 20, 50, 80]) == [30, 80]


def test_track_history_retains_a_single_identity_over_time() -> None:
    history = {}
    observation = {
        "track_id": 7,
        "class_name": "person",
        "confidence": 0.91,
        "bbox": [10, 20, 50, 80],
    }

    update_track_history(history, observation, timestamp_seconds=1.0)
    observation["bbox"] = [20, 30, 60, 90]
    update_track_history(history, observation, timestamp_seconds=2.5)

    assert list(history) == [7]
    assert history[7]["first_seen"] == 1.0
    assert history[7]["last_seen"] == 2.5
    assert history[7]["positions"] == [
        {"timestamp_seconds": 1.0, "position": [30, 80], "bbox": [10, 20, 50, 80]},
        {"timestamp_seconds": 2.5, "position": [40, 90], "bbox": [20, 30, 60, 90]},
    ]
    assert history[7]["zones"] == []
    assert history[7]["events"] == []


def test_draw_tracks_returns_an_annotated_copy() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    tracks = [
        {
            "track_id": 7,
            "class_name": "person",
            "confidence": 0.91,
            "bbox": [20, 20, 80, 90],
        }
    ]

    annotated = ObjectTracker.draw_tracks(frame, tracks)

    assert annotated.shape == frame.shape
    assert np.array_equal(frame, np.zeros((100, 160, 3), dtype=np.uint8))
    assert not np.array_equal(annotated, frame)
