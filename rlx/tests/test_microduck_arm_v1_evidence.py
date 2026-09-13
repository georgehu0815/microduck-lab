import pytest

from microduck_arm_v1.evidence import require_current
from microduck_arm_v1.validation import provenance


def test_current_failed_simulation_is_valid_evidence():
    require_current({"provenance": provenance(), "hardware_ready": False, "simulation_only": True, "success": False})


def test_stale_sources_are_rejected():
    with pytest.raises(ValueError, match="provenance"):
        require_current({"provenance": {}})


def test_physical_claim_is_rejected():
    with pytest.raises(ValueError, match="simulation-only"):
        require_current({"provenance": provenance(), "hardware_ready": True, "simulation_only": True, "success": True})
