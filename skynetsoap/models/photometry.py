import astropy.io.fits as fits
import astropy.table as tbl

# stop Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

class Photometry:
    def __init__(self, ra, dec, images):
        self.ra = ra
        self.dec = dec
        self.images = images
        self.results = None

    def open_fits(self, image):
        """Open an image using astropy."""
        hdulist = fits.open(image)
        image_data = hdulist[0].data
        header = hdulist[0].header
        hdulist.close()
        return image_data, header
    
    def run_aperture_photometry(self):
        """Run aperture photometry on the downloaded images."""
        phot_table = tbl.Table(names=('image', 'telescope', 'filter', 'exp_len', 'mjd', 'xcenter', 'ycenter', 'flux', 'flux_err', 'normalized_flux', 'normalized_flux_err', 'magnitude', 'magnitude_err'))
        return
            