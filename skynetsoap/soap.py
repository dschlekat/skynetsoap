from models.observation import Observation
from models.photometry import Photometry
from models.plotter import Plotter
from models.table import Table

class Soap:
    def __init__(self, ra, dec, observation_id, calibrate=False):
        self.ra = ra
        self.dec = dec
        self.observation_id = observation_id
        self.observation = Observation(observation_id)
        self.photometry = None
        self.table = None
        self.plotter = None
        self.calibrate = calibrate

    def download_images(self):
        """Download images using the Skynet API."""
        self.observation.download_images()
        print(f"Downloaded images for observation {self.observation_id}.")

    def perform_photometry(self):
        """Conduct aperture photometry on the downloaded images."""
        self.photometry = Photometry(self.ra, self.dec, self.observation.images)
        self.photometry.run_aperture_photometry()
        print("Aperture photometry complete.")

        # Optional: Perform magnitude calibration if a catalog is provided
        if self.catalog:
            self.photometry.calibrate_magnitudes(self.catalog)
            print("Magnitude calibration complete.")

    def generate_table(self, filetype="csv"):
        """Generate and return a photometric table."""
        self.table = Table(self.photometry.results)
        return self.table.create_table(filetype)

    def generate_plot(self, units="flux", normalize=True):
        """Generate and return a flux vs time plot."""
        self.plotter = Plotter(self.photometry.results)
        return self.plotter.plot_flux_vs_time(units=units, normalize=normalize)