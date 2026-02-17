"""Unit tests for skynetsoap.astrometry.hybrid_solver."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from astropy.wcs import WCS

from skynetsoap.astrometry.hybrid_solver import HybridAstrometryNetSolver


class TestHybridAstrometryNetSolver:
    """Tests for HybridAstrometryNetSolver."""

    def test_initialization_defaults(self):
        """Verify default initialization."""
        solver = HybridAstrometryNetSolver()
        assert solver.mode == "auto"
        assert solver.fallback_enabled is True
        assert solver._local_solver is None
        assert solver._api_solver is None

    def test_initialization_with_params(self):
        """Verify initialization with custom parameters."""
        local_config = {"timeout": 120, "scale_low": 0.5}
        api_config = {"timeout": 300}

        solver = HybridAstrometryNetSolver(
            mode="local",
            local_config=local_config,
            api_config=api_config,
            fallback_enabled=False,
        )

        assert solver.mode == "local"
        assert solver.fallback_enabled is False
        assert solver._local_config == local_config
        assert solver._api_config == api_config

    def test_invalid_mode_raises_error(self):
        """Verify ValueError for invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            HybridAstrometryNetSolver(mode="invalid_mode")

    def test_valid_modes(self):
        """Verify all valid modes can be initialized."""
        valid_modes = ["auto", "local", "api", "local_only", "api_only"]
        for mode in valid_modes:
            solver = HybridAstrometryNetSolver(mode=mode)
            assert solver.mode == mode

    def test_is_solved_with_valid_wcs(self, mock_image_with_wcs):
        """Verify is_solved() returns True for image with valid WCS."""
        solver = HybridAstrometryNetSolver()
        assert solver.is_solved(mock_image_with_wcs) is True

    def test_is_solved_without_wcs(self, mock_image_no_wcs):
        """Verify is_solved() returns False for image without WCS."""
        solver = HybridAstrometryNetSolver()
        assert solver.is_solved(mock_image_no_wcs) is False

    def test_auto_mode_prefers_local_when_available(self, mock_image_no_wcs):
        """Verify auto mode tries local first when available."""
        solver = HybridAstrometryNetSolver(mode="auto")

        mock_wcs = Mock(spec=WCS)

        with patch(
            "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver.is_available",
            return_value=True,
        ):
            with patch.object(solver, "_get_local_solver") as mock_get_local:
                mock_local = Mock()
                mock_local.solve.return_value = mock_wcs
                mock_get_local.return_value = mock_local

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs == mock_wcs
                mock_local.solve.assert_called_once_with(mock_image_no_wcs)

    def test_auto_mode_uses_api_when_local_unavailable(self, mock_image_no_wcs):
        """Verify auto mode uses API when local not available."""
        solver = HybridAstrometryNetSolver(mode="auto")

        mock_wcs = Mock(spec=WCS)

        with patch(
            "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver.is_available",
            return_value=False,
        ):
            with patch.object(solver, "_get_api_solver") as mock_get_api:
                mock_api = Mock()
                mock_api.solve.return_value = mock_wcs
                mock_get_api.return_value = mock_api

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs == mock_wcs
                mock_api.solve.assert_called_once_with(mock_image_no_wcs)

    def test_local_mode_with_fallback(self, mock_image_no_wcs):
        """Verify local mode falls back to API on failure."""
        solver = HybridAstrometryNetSolver(mode="local", fallback_enabled=True)

        mock_wcs = Mock(spec=WCS)

        with patch.object(solver, "_get_local_solver") as mock_get_local:
            with patch.object(solver, "_get_api_solver") as mock_get_api:
                # Local solver fails
                mock_local = Mock()
                mock_local.solve.return_value = None
                mock_get_local.return_value = mock_local

                # API solver succeeds
                mock_api = Mock()
                mock_api.solve.return_value = mock_wcs
                mock_get_api.return_value = mock_api

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs == mock_wcs
                mock_local.solve.assert_called_once()
                mock_api.solve.assert_called_once()

    def test_local_only_mode_no_fallback(self, mock_image_no_wcs):
        """Verify local_only mode does not try API."""
        solver = HybridAstrometryNetSolver(mode="local_only")

        with patch.object(solver, "_get_local_solver") as mock_get_local:
            with patch.object(solver, "_get_api_solver") as mock_get_api:
                # Local solver fails
                mock_local = Mock()
                mock_local.solve.return_value = None
                mock_get_local.return_value = mock_local

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs is None
                mock_local.solve.assert_called_once()
                mock_get_api.assert_not_called()

    def test_api_only_mode(self, mock_image_no_wcs):
        """Verify api_only mode only uses API solver."""
        solver = HybridAstrometryNetSolver(mode="api_only")

        mock_wcs = Mock(spec=WCS)

        with patch.object(solver, "_get_api_solver") as mock_get_api:
            with patch.object(solver, "_get_local_solver") as mock_get_local:
                mock_api = Mock()
                mock_api.solve.return_value = mock_wcs
                mock_get_api.return_value = mock_api

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs == mock_wcs
                mock_api.solve.assert_called_once()
                mock_get_local.assert_not_called()

    def test_api_mode_with_local_fallback(self, mock_image_no_wcs):
        """Verify api mode falls back to local on failure."""
        solver = HybridAstrometryNetSolver(mode="api", fallback_enabled=True)

        mock_wcs = Mock(spec=WCS)

        with patch.object(solver, "_get_api_solver") as mock_get_api:
            with patch.object(solver, "_get_local_solver") as mock_get_local:
                # API solver fails
                mock_api = Mock()
                mock_api.solve.return_value = None
                mock_get_api.return_value = mock_api

                # Local solver succeeds
                mock_local = Mock()
                mock_local.solve.return_value = mock_wcs
                mock_get_local.return_value = mock_local

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs == mock_wcs
                mock_api.solve.assert_called_once()
                mock_local.solve.assert_called_once()

    def test_fallback_disabled(self, mock_image_no_wcs):
        """Verify fallback_enabled=False prevents secondary solver."""
        solver = HybridAstrometryNetSolver(mode="local", fallback_enabled=False)

        with patch.object(solver, "_get_local_solver") as mock_get_local:
            with patch.object(solver, "_get_api_solver") as mock_get_api:
                # Local solver fails
                mock_local = Mock()
                mock_local.solve.return_value = None
                mock_get_local.return_value = mock_local

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs is None
                mock_local.solve.assert_called_once()
                mock_get_api.assert_not_called()

    def test_both_solvers_fail(self, mock_image_no_wcs):
        """Verify None returned when all solvers fail."""
        solver = HybridAstrometryNetSolver(mode="local", fallback_enabled=True)

        with patch.object(solver, "_get_local_solver") as mock_get_local:
            with patch.object(solver, "_get_api_solver") as mock_get_api:
                # Both solvers fail
                mock_local = Mock()
                mock_local.solve.return_value = None
                mock_get_local.return_value = mock_local

                mock_api = Mock()
                mock_api.solve.return_value = None
                mock_get_api.return_value = mock_api

                wcs = solver.solve(mock_image_no_wcs)

                assert wcs is None
                mock_local.solve.assert_called_once()
                mock_api.solve.assert_called_once()

    def test_solve_already_solved_image(self, mock_image_with_wcs):
        """Verify early return for already-solved image."""
        solver = HybridAstrometryNetSolver()

        with patch.object(solver, "_get_local_solver") as mock_get_local:
            with patch.object(solver, "_get_api_solver") as mock_get_api:
                wcs = solver.solve(mock_image_with_wcs)

                assert wcs is not None
                # Should not call any solver
                mock_get_local.assert_not_called()
                mock_get_api.assert_not_called()

    def test_get_local_solver_lazy_initialization(self):
        """Verify local solver is lazy-initialized."""
        solver = HybridAstrometryNetSolver()

        assert solver._local_solver is None

        with patch(
            "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver.is_available",
            return_value=True,
        ):
            with patch(
                "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver"
            ) as mock_cls:
                mock_instance = Mock()
                mock_cls.return_value = mock_instance

                local_solver = solver._get_local_solver()

                assert local_solver == mock_instance
                assert solver._local_solver == mock_instance
                mock_cls.assert_called_once()

                # Second call should return cached instance
                local_solver2 = solver._get_local_solver()
                assert local_solver2 == mock_instance
                assert mock_cls.call_count == 1  # Not called again

    def test_get_local_solver_returns_none_when_unavailable(self):
        """Verify _get_local_solver returns None when solve-field not available."""
        solver = HybridAstrometryNetSolver()

        with patch(
            "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver.is_available",
            return_value=False,
        ):
            local_solver = solver._get_local_solver()
            assert local_solver is None

    def test_get_local_solver_handles_initialization_error(self):
        """Verify _get_local_solver handles FileNotFoundError gracefully."""
        solver = HybridAstrometryNetSolver()

        with patch(
            "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver.is_available",
            return_value=True,
        ):
            with patch(
                "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver",
                side_effect=FileNotFoundError("solve-field not found"),
            ):
                local_solver = solver._get_local_solver()
                assert local_solver is None

    def test_get_api_solver_lazy_initialization(self):
        """Verify API solver is lazy-initialized."""
        solver = HybridAstrometryNetSolver()

        assert solver._api_solver is None

        with patch(
            "skynetsoap.astrometry.hybrid_solver.AstrometryNetSolver"
        ) as mock_cls:
            mock_instance = Mock()
            mock_cls.return_value = mock_instance

            api_solver = solver._get_api_solver()

            assert api_solver == mock_instance
            assert solver._api_solver == mock_instance
            mock_cls.assert_called_once()

            # Second call should return cached instance
            api_solver2 = solver._get_api_solver()
            assert api_solver2 == mock_instance
            assert mock_cls.call_count == 1  # Not called again

    def test_determine_solver_priority_auto_local_available(self):
        """Verify auto mode priority when local is available."""
        solver = HybridAstrometryNetSolver(mode="auto")

        with patch(
            "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver.is_available",
            return_value=True,
        ):
            primary, secondary = solver._determine_solver_priority()
            assert primary == "local"
            assert secondary == "api"

    def test_determine_solver_priority_auto_local_unavailable(self):
        """Verify auto mode priority when local is unavailable."""
        solver = HybridAstrometryNetSolver(mode="auto")

        with patch(
            "skynetsoap.astrometry.hybrid_solver.LocalAstrometryNetSolver.is_available",
            return_value=False,
        ):
            primary, secondary = solver._determine_solver_priority()
            assert primary == "api"
            assert secondary == "local"

    def test_determine_solver_priority_all_modes(self):
        """Verify solver priority for all modes."""
        test_cases = [
            ("local", ("local", "api")),
            ("api", ("api", "local")),
            ("local_only", ("local", None)),
            ("api_only", ("api", None)),
        ]

        for mode, expected in test_cases:
            solver = HybridAstrometryNetSolver(mode=mode)
            result = solver._determine_solver_priority()
            assert result == expected, f"Failed for mode {mode}"

    def test_try_solver_exception_handling(self, mock_image_no_wcs):
        """Verify exception handling in _try_solver."""
        solver = HybridAstrometryNetSolver()

        with patch.object(solver, "_get_local_solver") as mock_get_local:
            mock_local = Mock()
            mock_local.solve.side_effect = Exception("Unexpected error")
            mock_get_local.return_value = mock_local

            wcs = solver._try_solver("local", mock_image_no_wcs)

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
