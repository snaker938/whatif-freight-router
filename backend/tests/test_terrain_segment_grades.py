from __future__ import annotations

import pytest

from app.settings import settings
from app.terrain_dem import segment_grade_profile


def test_segment_grade_profile_is_deterministic_and_aligned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "terrain_allow_synthetic_grid", True)
    coords = [(-2.2, 53.3), (-1.7, 52.9), (-1.2, 52.3), (-0.6, 51.8)]
    segment_distances = [11_000.0, 14_000.0, 16_000.0]
    first = segment_grade_profile(
        coordinates_lon_lat=coords,
        segment_distances_m=segment_distances,
    )
    second = segment_grade_profile(
        coordinates_lon_lat=coords,
        segment_distances_m=segment_distances,
    )

    assert len(first) == len(segment_distances)
    assert first == second
    assert any(abs(grade) > 1e-6 for grade in first)
    assert all(-0.25 <= grade <= 0.25 for grade in first)
