import os

from .models.observation import Observation
from .models.photometry import Photometry
from .models.plotter import Plotter
from .models.table import Table

class Soap:
    def __init__(self, ra, dec, observation_id, calibrate=False, target_name=None):
        self.ra = ra
        self.dec = dec
        self.observation_id = observation_id
        self.observation = Observation(observation_id)
        self.photometry = None
        self.table = None
        self.plotter = None
        self.calibrate = calibrate
        self.target_name = target_name

        if not os.path.exists("soap_images"):
            os.makedirs("soap_images")
        if not os.path.exists("soap_results"):
            os.makedirs("soap_results")


    def download_images(self):
        """Download images using the Skynet API."""
        name = self.observation.download_images()
        if self.target_name is not None:
            name = self.target_name
        else:
            self.target_name = name
        print(f"Downloaded images for observation {self.observation_id} ({name}).")

    def perform_photometry(self):
        """Conduct aperture photometry on the downloaded images."""
        self.photometry = Photometry(self.ra, self.dec, self.observation.images)
        self.photometry.run_aperture_photometry()
        print("Aperture photometry complete.")

        # Optional: Perform magnitude calibration if a catalog is provided
        if self.calibrate:
            self.photometry.calibrate_magnitudes()
            print("Magnitude calibration complete.")

    def generate_table(self, filetype="csv", path="soap_results/photometry_table"):
        """Generate and return a photometric table."""
        self.table = Table(self.photometry.results)
        return self.table.create_table(filetype, path)

    def generate_plot(self, units="flux", path="soap_results/photometry_plot"):
        """Generate and return a flux vs time plot."""
        self.plotter = Plotter(self.photometry.results)
        return self.plotter.create_plot(units, path)