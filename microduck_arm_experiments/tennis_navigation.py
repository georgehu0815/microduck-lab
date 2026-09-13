"""Offline simulator-truth route screening; not an on-robot controller."""

from __future__ import annotations

from dataclasses import asdict

from .tennis_controller import NavigationProfile
from .tennis_release import FORECAST_PROVENANCE, _clone_environment, _terminal_info


NAVIGATION_PROFILES = (
    NavigationProfile(),
    NavigationProfile(warmup_s=1.5),
    NavigationProfile(warmup_s=3.25),
    NavigationProfile(warmup_s=2.5),
    NavigationProfile(cruise_m_s=.20, slow_m_s=.178),
    NavigationProfile(gain_scale=1.08),
    NavigationProfile(cruise_m_s=.16, slow_m_s=.14),
    NavigationProfile(warmup_s=1.),
    NavigationProfile(warmup_s=3.),
    NavigationProfile(cruise_m_s=.22, slow_m_s=.196),
    NavigationProfile(waypoint_y_m=.13),
    NavigationProfile(gain_scale=.95),
    NavigationProfile(cruise_m_s=.20, slow_m_s=.178, gain_scale=1.1),
    NavigationProfile(cruise_m_s=.16, slow_m_s=.142, gain_scale=1.1),
    NavigationProfile(waypoint_y_m=.09),
)


def forecast_navigation(environment, profile):
    clone = _clone_environment(environment)
    latest = {}
    try:
        clone.controller.navigation_forecasting = True
        clone.controller.navigator.profile = profile
        while not clone.done:
            _, _, latest = clone.step(clone.teacher_action())
        return {
            **FORECAST_PROVENANCE,
            **_terminal_info(clone, latest),
            "profile": asdict(profile),
            "start_step": environment.steps,
            "terminal_step": clone.steps,
            "forecast_steps": clone.steps - environment.steps,
        }
    except Exception as error:
        return {
            **FORECAST_PROVENANCE,
            "success": False,
            "failure_reason": "navigation_forecast_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "profile": asdict(profile),
        }
    finally:
        clone.close()


def select_navigation_profile(environment):
    evidence = []
    for profile in NAVIGATION_PROFILES:
        record = forecast_navigation(environment, profile)
        evidence.append(record)
        if record["success"]:
            return profile, evidence
    return None, evidence
