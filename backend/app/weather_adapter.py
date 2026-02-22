from __future__ import annotations

from .models import WeatherImpactConfig

_PROFILE_SPEED_MULTIPLIER: dict[str, float] = {
    "clear": 1.00,
    "rain": 1.08,
    "storm": 1.20,
    "snow": 1.28,
    "fog": 1.12,
}

_PROFILE_INCIDENT_MULTIPLIER: dict[str, float] = {
    "clear": 1.00,
    "rain": 1.15,
    "storm": 1.50,
    "snow": 1.80,
    "fog": 1.25,
}


def _clamp_intensity(value: float) -> float:
    return max(0.0, min(2.0, float(value)))


def _scaled_multiplier(profile_base: float, intensity: float) -> float:
    if profile_base <= 1.0:
        return 1.0
    return 1.0 + ((profile_base - 1.0) * intensity)


def weather_speed_multiplier(config: WeatherImpactConfig) -> float:
    if not config.enabled:
        return 1.0
    intensity = _clamp_intensity(config.intensity)
    base = _PROFILE_SPEED_MULTIPLIER.get(config.profile, 1.0)
    return max(0.5, _scaled_multiplier(base, intensity))


def weather_incident_multiplier(config: WeatherImpactConfig) -> float:
    if not config.enabled or not config.apply_incident_uplift:
        return 1.0
    intensity = _clamp_intensity(config.intensity)
    base = _PROFILE_INCIDENT_MULTIPLIER.get(config.profile, 1.0)
    return max(0.5, _scaled_multiplier(base, intensity))


def weather_summary(config: WeatherImpactConfig) -> dict[str, float | str | bool]:
    return {
        "enabled": bool(config.enabled),
        "profile": config.profile,
        "intensity": _clamp_intensity(config.intensity),
        "apply_incident_uplift": bool(config.apply_incident_uplift),
        "speed_multiplier": round(weather_speed_multiplier(config), 6),
        "incident_multiplier": round(weather_incident_multiplier(config), 6),
    }

