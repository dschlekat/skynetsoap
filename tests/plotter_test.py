import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skynetsoap.models.plotter import Plotter

class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"mjd": 59000, "flux": 1.0, "normalized_flux": 0.9, "magnitude": 15.0, "error": 0.1},
            {"mjd": 59001, "flux": 1.1, "normalized_flux": 0.95, "magnitude": 14.9, "error": 0.1},
            {"mjd": 59002, "flux": 0.9, "normalized_flux": 0.85, "magnitude": 15.1, "error": 0.1}
        ]
        self.plotter = Plotter(self.results)

    def tearDown(self):
        if os.path.exists("soap_results"):
            shutil.rmtree("soap_results")

    def test_create_plot_flux(self):
        self.plotter.units = "flux"
        self.plotter.create_plot()
        self.assertTrue(os.path.exists("soap_results/photometry_plot.png"))

    def test_create_plot_normalized_flux(self):
        self.plotter.units = "normalized_flux"
        self.plotter.create_plot(path="soap_results/normalized_flux_plot")
        self.assertTrue(os.path.exists("soap_results/normalized_flux_plot.png"))

    def test_create_plot_magnitude(self):
        self.plotter.units = "magnitude"
        self.plotter.create_plot(path="soap_results/magnitude_plot")
        self.assertTrue(os.path.exists("soap_results/magnitude_plot.png"))

if __name__ == '__main__':
    unittest.main()