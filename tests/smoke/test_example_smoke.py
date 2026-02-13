"""Opt-in smoke test based on the previous example.py workflow."""

from __future__ import annotations

import pytest
from astropy.coordinates import SkyCoord

from skynetsoap import Soap


pytestmark = [
    pytest.mark.smoke,
]


@pytest.mark.filterwarnings("ignore::astropy.wcs.wcs.FITSFixedWarning")
def test_example_observations_smoke(tmp_path):
    """Run the two example observation workflows end-to-end."""
    # JC filter calibration example
    soap_jc = Soap(
        observation_id=13110404,
        verbose=True,
        image_dir=str(tmp_path / "soap_images"),
        result_dir=str(tmp_path / "soap_results"),
    )
    soap_jc.download()
    result_jc = soap_jc.run()
    export_jc = soap_jc.export()

    assert len(result_jc) > 0
    assert export_jc.exists()
    assert "limiting_mag" in result_jc.table.colnames

    target = SkyCoord(
        "03:27:48.9394738008 +74:39:52.531563600",
        unit=("hourangle", "deg"),
    )
    target_result = result_jc.extract_target(target, radius_arcsec=3.0)
    target_export = target_result.export()
    assert target_export.exists()

    # SDSS filter calibration example
    soap_sdss = Soap(
        observation_id=12313253,
        verbose=True,
        image_dir=str(tmp_path / "soap_images"),
        result_dir=str(tmp_path / "soap_results"),
    )
    soap_sdss.download()
    result_sdss = soap_sdss.run()
    export_sdss = soap_sdss.export()

    assert len(result_sdss) > 0
    assert export_sdss.exists()
    assert "limiting_mag" in result_sdss.table.colnames
