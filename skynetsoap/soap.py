import os

from .models.observation import Observation
from .models.photometry import Photometry
from .models.plotter import Plotter
from .models.table import Table

class Soap:
    def __init__(self, ra, dec, observation_id, target_name, calibrate=False, image_dir="soap_images", result_dir="soap_results", pickle=True):
        if isinstance(ra, str):
            try: 
                if ":" in ra:
                    ra = float(ra.split(":")[0])*15 + float(ra.split(":")[1])/4 + float(ra.split(":")[2])/240
                elif " " in ra:
                    ra = float(ra.split(" ")[0])*15 + float(ra.split(" ")[1])/4 + float(ra.split(" ")[2])/240
                else:
                    raise ValueError
            except:
                pass
        if isinstance(dec, str):
            try:
                if ":" in dec:
                    dec = float(dec.split(":")[0]) + float(dec.split(":")[1])/60 + float(dec.split(":")[2])/3600
                elif " " in dec:
                    dec = float(dec.split(" ")[0]) + float(dec.split(" ")[1])/60 + float(dec.split(" ")[2])/3600
                else:
                    raise ValueError
            except:
                pass
        
        self.ra = ra
        self.dec = dec
        self.observation_id = observation_id
        self.target_name = target_name
        self.observation = Observation(observation_id)
        self.photometry = None
        self.table = None
        self.plotter = None
        self.calibrate = calibrate

        self.img_path = f"{image_dir}/{target_name}"
        self.res_path = f"{result_dir}/{target_name}"

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)


    def download_images(self):
        """Download images using the Skynet API."""
        self.observation.get_obs()
        name = self.observation.download_images(path=self.img_path)
        print(f"Downloaded images for Skynet Observation {self.observation_id} ({name}).")

    def photometry_pipeline(self):
        """Conduct aperture photometry on the downloaded images."""
        self.photometry = Photometry(self.ra, self.dec, self.observation.images, self.img_path)
        self.photometry.run_pipeline()
        print("Aperture photometry complete.")

        # Optional: Perform magnitude calibration if a catalog is provided
        if self.calibrate:
            self.photometry.calibrate_magnitudes()
            print("Magnitude calibration complete.")

    def generate_table(self, filetype="csv"):
        """Generate and return a photometric table."""
        self.table = Table(self.photometry.results)
        print(f"Table saved to {self.res_path}.")
        return self.table.create_table(filetype, self.res_path)

    def generate_plot(self, units="flux"):
        """Generate and return a flux vs time plot."""
        self.plotter = Plotter(self.photometry.results, units=units)
        print(f"Plot saved to {self.res_path}.")
        return self.plotter.create_plot(self.res_path)
    
    def calibrate_magnitudes(self):
        """Calibrate magnitudes using a catalog."""
        if self.calibrate:
            print("Magnitude calibration already complete.")
        else:
            self.calibrate = True
            self.photometry.calibrate_magnitudes()
            print("Magnitude calibration complete.")
        return
    
    def run_all(self, units="flux", filetype="csv"):
        """Run the entire SOAP pipeline."""
        self.download_images()
        self.photometry_pipeline()
        self.generate_table(filetype=filetype)
        self.generate_plot(units=units)
        print("SOAP pipeline complete.")