"""Unit tests for skynetsoap.astrometry.local_solver."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from astropy.io import fits
from astropy.wcs import WCS

from skynetsoap.astrometry.local_solver import LocalAstrometryNetSolver


class TestLocalAstrometryNetSolver:
    """Tests for LocalAstrometryNetSolver."""

    def test_is_available_when_installed(self):
        """Verify is_available() returns True when solve-field is in PATH."""
        with patch("shutil.which", return_value="/usr/local/bin/solve-field"):
            assert LocalAstrometryNetSolver.is_available() is True

    def test_is_available_when_not_installed(self):
        """Verify is_available() returns False when solve-field is not found."""
        with patch("shutil.which", return_value=None):
            assert LocalAstrometryNetSolver.is_available() is False

    def test_binary_path_validation_absolute_path_exists(self):
        """Verify custom binary path validation for existing file."""
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
            tmp_path = Path(tmp.name)
            tmp_path.chmod(0o755)  # Make executable

        try:
            solver = LocalAstrometryNetSolver(binary_path=str(tmp_path))
            assert solver.binary_path == str(tmp_path)
        finally:
            tmp_path.unlink()

    def test_binary_path_validation_not_found(self):
        """Verify FileNotFoundError when binary not found."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="not found in PATH"):
                LocalAstrometryNetSolver(binary_path="nonexistent-binary")

    def test_binary_path_validation_absolute_nonexistent(self):
        """Verify FileNotFoundError for nonexistent absolute path."""
        with pytest.raises(FileNotFoundError, match="not found at"):
            LocalAstrometryNetSolver(binary_path="/nonexistent/path/solve-field")

    def test_initialization_with_defaults(self):
        """Verify default parameter initialization."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()
            assert solver.timeout == 300
            assert solver.scale_low is None
            assert solver.scale_high is None
            assert solver.depth is None
            assert solver.downsample is None
            assert solver.extra_args == []

    def test_initialization_with_custom_params(self):
        """Verify initialization with custom parameters."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver(
                timeout=120,
                scale_low=0.5,
                scale_high=2.0,
                depth=[20, 40, 60],
                downsample=2,
                extra_args=["--no-verify"],
            )
            assert solver.timeout == 120
            assert solver.scale_low == 0.5
            assert solver.scale_high == 2.0
            assert solver.depth == [20, 40, 60]
            assert solver.downsample == 2
            assert solver.extra_args == ["--no-verify"]

    def test_is_solved_with_valid_wcs(self, mock_image_with_wcs):
        """Verify is_solved() returns True for image with valid WCS."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()
            assert solver.is_solved(mock_image_with_wcs) is True

    def test_is_solved_without_wcs(self, mock_image_no_wcs):
        """Verify is_solved() returns False for image without WCS."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()
            assert solver.is_solved(mock_image_no_wcs) is False

    def test_build_command_minimal(self):
        """Verify command construction with minimal parameters."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()
            image_path = Path("/path/to/image.fits")
            output_dir = Path("/tmp/astrom")
            basename = "image"

            cmd = solver._build_command(image_path, output_dir, basename)

            assert cmd[0] == "/usr/bin/solve-field"
            assert "--no-plots" in cmd
            assert "--overwrite" in cmd
            assert "--dir" in cmd
            assert str(output_dir) in cmd
            assert "--wcs" in cmd
            assert str(output_dir / "image.wcs") in cmd
            assert str(image_path) in cmd

    def test_build_command_with_scale_hints(self):
        """Verify command construction with scale hints."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver(scale_low=0.5, scale_high=2.0)
            image_path = Path("/path/to/image.fits")
            output_dir = Path("/tmp/astrom")
            basename = "image"

            cmd = solver._build_command(image_path, output_dir, basename)

            assert "--scale-low" in cmd
            assert "0.5" in cmd
            assert "--scale-high" in cmd
            assert "2.0" in cmd
            assert "--scale-units" in cmd
            assert "arcsecperpix" in cmd

    def test_build_command_with_depth_single(self):
        """Verify command construction with single depth value."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver(depth=50)
            cmd = solver._build_command(Path("img.fits"), Path("/tmp"), "img")

            assert "--depth" in cmd
            assert "50" in cmd

    def test_build_command_with_depth_list(self):
        """Verify command construction with depth list."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver(depth=[20, 40, 60])
            cmd = solver._build_command(Path("img.fits"), Path("/tmp"), "img")

            assert "--depth" in cmd
            assert "20,40,60" in cmd

    def test_build_command_with_downsample(self):
        """Verify command construction with downsample factor."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver(downsample=2)
            cmd = solver._build_command(Path("img.fits"), Path("/tmp"), "img")

            assert "--downsample" in cmd
            assert "2" in cmd

    def test_build_command_with_extra_args(self):
        """Verify command construction with extra arguments."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver(
                extra_args=["--no-verify", "--cpulimit", "60"]
            )
            cmd = solver._build_command(Path("img.fits"), Path("/tmp"), "img")

            assert "--no-verify" in cmd
            assert "--cpulimit" in cmd
            assert "60" in cmd

    def test_parse_wcs_from_output_success(self):
        """Verify WCS parsing from .wcs file."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()

            # Create a temporary WCS file
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                wcs_file = tmpdir / "test.wcs"

                # Create a simple WCS header and write it
                wcs = WCS(naxis=2)
                wcs.wcs.crpix = [128, 128]
                wcs.wcs.crval = [51.95, 74.66]
                wcs.wcs.cdelt = [-0.0002777777778, 0.0002777777778]
                wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

                hdu = fits.PrimaryHDU(header=wcs.to_header())
                hdu.writeto(wcs_file)

                # Parse it
                parsed_wcs = solver._parse_wcs_from_output(tmpdir, "test")

                assert parsed_wcs is not None
                assert isinstance(parsed_wcs, WCS)

    def test_parse_wcs_from_output_file_not_found(self):
        """Verify None returned when .wcs file doesn't exist."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()

            with tempfile.TemporaryDirectory() as tmpdir:
                parsed_wcs = solver._parse_wcs_from_output(Path(tmpdir), "nonexistent")
                assert parsed_wcs is None

    def test_solve_already_solved_image(self, mock_image_with_wcs):
        """Verify early return for already-solved image."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()
            wcs = solver.solve(mock_image_with_wcs)

            assert wcs is not None
            # Should return existing WCS without calling subprocess

    def test_solve_timeout(self, mock_image_no_wcs):
        """Verify timeout handling during solve."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver(timeout=1)

            with patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)
            ):
                wcs = solver.solve(mock_image_no_wcs)

                assert wcs is None

    def test_solve_nonzero_return_code(self, mock_image_no_wcs):
        """Verify handling of solve-field failure."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()

            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stderr = "Error: could not solve field"

            with patch("subprocess.run", return_value=mock_result):
                wcs = solver.solve(mock_image_no_wcs)

                assert wcs is None

    def test_solve_exception_handling(self, mock_image_no_wcs):
        """Verify exception handling during solve."""
        with patch("shutil.which", return_value="/usr/bin/solve-field"):
            solver = LocalAstrometryNetSolver()

            with patch("subprocess.run", side_effect=Exception("Unexpected error")):
                wcs = solver.solve(mock_image_no_wcs)

                assert wcs is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_image_with_wcs():
    """Mock FITSImage with valid WCS."""
    image = MagicMock()
    image.path = Path("test_with_wcs.fits")
    image.header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 128.0,
        "CRPIX2": 128.0,
        "CRVAL1": 51.95,
        "CRVAL2": 74.66,
    }
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [128, 128]
    wcs.wcs.crval = [51.95, 74.66]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    image.wcs = wcs
    return image


@pytest.fixture
def mock_image_no_wcs():
    """Mock FITSImage without WCS."""
    image = MagicMock()
    image.path = Path("test_no_wcs.fits")
    image.header = {"SIMPLE": True, "BITPIX": -32}  # No CTYPE keywords
    image.wcs = None
    return image
