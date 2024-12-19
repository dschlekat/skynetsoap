import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from skynetsoap.models.plotter import Plotter

class TestPlotter(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"mjd": 59000, "flux": 10.0, "flux_err": 0.03, "normalized_flux": 0.9, "normalized_flux_err": 0.03, "magnitude": 15.0, "magnitude_err": 0.03 },
            {"mjd": 59001, "flux": 10.1, "flux_err": 0.03, "normalized_flux": 0.95, "normalized_flux_err": 0.03, "magnitude": 14.9, "magnitude_err": 0.03 },
            {"mjd": 59002, "flux": 9.9, "flux_err": 0.03, "normalized_flux": 0.85, "normalized_flux_err": 0.03, "magnitude": 15.1, "magnitude_err": 0.03 }
        ]
        self.plotter = Plotter(self.results)

    def tearDown(self):
        if os.path.exists("soap_results"):
            shutil.rmtree("soap_results")

    def test_create_plot_flux(self):
        self.plotter.units = "flux"
        self.plotter.create_plot()
        self.assertTrue(os.path.exists("soap_results/flux_plot.png"))

    def test_create_plot_normalized_flux(self):
        self.plotter.units = "normalized_flux"
        self.plotter.create_plot()
        self.assertTrue(os.path.exists("soap_results/normalized_flux_plot.png"))

    def test_create_plot_magnitude(self):
        self.plotter.units = "magnitude"
        self.plotter.create_plot()
        self.assertTrue(os.path.exists("soap_results/magnitude_plot.png"))

if __name__ == '__main__':
    unittest.main()