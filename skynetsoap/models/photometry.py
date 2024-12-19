import astropy.io.fits as fits
import astropy.table as tbl
from astropy.stats import SigmaClip, mad_std
import astropy.wcs as awcs
import numpy as np
from photutils.background import Background2D, MedianBackground
from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture, CircularAnnulus
from photutils.utils import calc_total_error
sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = MedianBackground()
from tqdm import tqdm

# stop Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# TODO: Add test
# TODO: Implement magnitude calibration
# TODO: Implement a better check for existing results
# TODO: Look into changing Table to QTable to retain units
# TODO: Implement variable aperture sizes

class Photometry:
    def __init__(self, ra, dec, images, path):
        self.target_ra = ra
        self.target_dec = dec
        self.images = images
        self.results = tbl.Table(names=(
            'image', 
            'telescope', 
            'filter', 
            'exp_len', 
            'mjd',
            'time_since_start',
            'xcenter', 
            'ycenter', 
            'flux', 
            'flux_err', 
            'normalized_flux', 
            'normalized_flux_err', 
            'magnitude', 
            'magnitude_err'
        ), dtype=(
            'U100', 
            'U100', 
            'U100', 
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
            'f8'
        ))
        self.path = path


    def open_fits(self, image):
        """Open an image using astropy."""
        img_path = f"{self.path}/{image}"
        hdulist = fits.open(img_path)
        data = hdulist[0].data
        header = hdulist[0].header
        hdulist.close()
        return data, header
    
    def calibrate_magnitudes(self):
        """Calibrate magnitudes using a catalog."""
        # TODO: Implement
        return
    
    def check_for_results(self):
        """Check if results already exist."""
        if len(self.results) > 0:
            return True
        # elif # FIXME: Pickle check
        #     return False
        else:
            return False

    def normalize_flux(self):
        # Create a lightcurve using the table data
        mjd = self.results['mjd']
        mjd_first = np.min(mjd)

        flux = self.results['flux']
        flux_err = self.results['flux_err']

        # create time from seconds since the first observation
        days_since = mjd - mjd_first
        seconds_since = days_since * 86400

        # Normalize the flux
        normalized_flux = flux / np.median(flux)
        normalized_flux_err = flux_err / np.median(flux)

        self.results['normalized_flux'] = normalized_flux
        self.results['normalized_flux_err'] = normalized_flux_err
        self.results['time_since_start'] = seconds_since
        return
    
    def sort_results(self):
        """Sort the results by MJD."""
        self.results.sort('mjd')
        return
    
    def run_pipeline(self):
        """Run aperture photometry on the downloaded images."""
        check = self.check_for_results()
        if check:
            print('Results already exist')
            return

        loop = tqdm(self.images)
        for image in loop:
            try:
                data, header = self.open_fits(image)
                loop.set_description('Processing %s' % image)
            except:
                loop.set_description('Error loading %s' % image)
                continue

            # Subtract the background
            bkg = Background2D(data, (32, 32), filter_size=(3, 3),
                                    sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            data -= bkg.background #background subtracted data
            rms_map = bkg.background_rms

            # get the wcs
            wcs = awcs.WCS(header)

            # Get the MJD from the wcs
            mjd = wcs.wcs.mjdobs

            sy, sx = data.shape
            xtarg, ytarg = wcs.all_world2pix(self.target_ra, self.target_dec, 0)
            pos_targ = np.transpose(([xtarg], [ytarg]))

            # Conduct aperture photometry
            aptarg = CircularAperture(pos_targ, r=5.5)
            antarg = CircularAnnulus(pos_targ, r_in=10, r_out=15)
            
            target_bkg_stats = ApertureStats(data, antarg, sigma_clip=sigma_clip)  #stats of the background annulus of the target
            target_bkg = target_bkg_stats.median * aptarg.area #estimated bkg flux of target aperture
            ann_rms = target_bkg_stats.std * np.sqrt(aptarg.area) #sigma of bkg flux of target aperture

            error = calc_total_error(data, rms_map, 1.0) 
            target_photometry = aperture_photometry(data, aptarg, error=error)
            target_photometry['aperture_sum_bkgsub'] = target_photometry['aperture_sum'] - target_bkg #background subtracted total flux of target

            # Save the data
            self.results.add_row([
                image, 
                header['TELESCOP'], 
                header['FILTER'], 
                header['EXPTIME'], 
                mjd,
                None, 
                xtarg, 
                ytarg, 
                target_photometry['aperture_sum_bkgsub'], 
                target_photometry['aperture_sum_err'], 
                None, 
                None, 
                None, 
                None
            ])

        self.normalize_flux()
        self.sort_results()
        return