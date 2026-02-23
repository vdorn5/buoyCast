import numpy as np

from src.physics_informed import deep_water_wavelength, clip_energy_physical


def test_deep_water_wavelength_reasonable_value():
    # For T=10 s, L ~ 9.81*100/(2*pi) ~ 156 m
    L = deep_water_wavelength(10.0)
    assert np.isfinite(L)
    assert 150.0 < float(L) < 170.0


def test_clip_energy_physical_enforces_nonnegativity_and_breaking_cap():
    # Negative energy should clip to 0
    clipped = clip_energy_physical([-1.0, 4.0], tp_s=[10.0, 10.0])
    assert clipped[0] == 0.0

    # Very large energy should be capped by steepness limit
    huge = clip_energy_physical([1e9], tp_s=[10.0])
    assert huge[0] < 1e9
