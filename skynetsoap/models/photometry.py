import astropy.io.fits as fits
import astropy.table as tbl
from astropy.stats import SigmaClip
import astropy.wcs as awcs
import numpy as np
from photutils.background import Background2D, MedianBackground
from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture, CircularAnnulus
from photutils.utils import calc_total_error
import os
from tqdm import tqdm

# stop Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# DONE: Needs to collect existing results and add to self.results
# TODO: 
# TODO: Add test
# TODO: Implement magnitude calibration
# TODO: Implement a better check for existing results
# TODO: Look into changing Table to QTable to retain units
# TODO: Implement variable aperture sizes

class Photometry:
    def __init__(self, ra, dec, images, img_dir, res_dir, sigma=3.0, bkg_estimator='median', aperture_radius=6, inner_annulus_radius=12, outer_annulus_radius=18):
        # Target coordinates
        self.target_ra = ra
        self.target_dec = dec

        # Photometry
        self.sigma_clip = SigmaClip(sigma=sigma)
        if bkg_estimator == 'median':
            self.bkg_estimator = MedianBackground()
        self.aperture_radius = aperture_radius  # in pixels
        self.inner_annulus_radius = inner_annulus_radius  # in pixels
        self.outer_annulus_radius = outer_annulus_radius  # in pixels

        # Data and metadata
        self.img_dir = img_dir
        self.res_dir = res_dir
        self.images = images
        self.results = tbl.Table(names=(
            'exp_number',
            'exp_name',            
            'telescope', 
            'filter', 
            'exp_len', 
            'mjd',
            'time_since_start',
            'xcenter', 
            'ycenter', 
            'ra_center',
            'dec_center',
            'flux', 
            'flux_err', 
            'normalized_flux', 
            'normalized_flux_err', 
            'magnitude', 
            'magnitude_err',
            'absolute_magnitude',
            'absolute_magnitude_err',
            'image_depth',
        ), dtype=(
            'i8',
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
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
            'f8',
        ))

    def check_for_results(self):
        """Check if results already exist."""
        if len(self.results) > 0:
            return True
        elif os.path.exists(f"{self.res_dir}/astropy_table.ecsv"): #FIXME needs to be more sophisticated
            return True
        else:
            return False

    def open_fits(self, image):
        """Open an image using astropy."""
        img_path = f"{self.img_dir}/{image}"
        hdulist = fits.open(img_path)
        data = hdulist[0].data
        header = hdulist[0].header
        hdulist.close()
        return data, header
    
    def calibrate_magnitudes(self):
        """Calibrate magnitudes using a catalog."""
        # TODO: Implement
        return
    
    def sort_results(self):
        """Sort the results by MJD."""
        self.results.sort('mjd')
        self.results['exp_number'] = np.arange(1, len(self.results)+1)
        return
    
    def run_pipeline(self):
        """Run aperture photometry on the downloaded images."""
        check = self.check_for_results()
        if check:
            print('Results already exist, reading from file.')
            self.results = tbl.Table.read(f"{self.res_dir}/astropy_table.ecsv")
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
                                    sigma_clip=self.sigma_clip, bkg_estimator=self.bkg_estimator)
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
            aptarg = CircularAperture(pos_targ, r=self.aperture_radius)  # aperture radius in pixels
            antarg = CircularAnnulus(pos_targ, r_in=self.inner_annulus_radius, r_out=self.outer_annulus_radius)

            target_bkg_stats = ApertureStats(data, antarg, sigma_clip=self.sigma_clip)  #stats of the background annulus of the target
            target_bkg = target_bkg_stats.median * aptarg.area #estimated bkg flux of target aperture
            ann_rms = target_bkg_stats.std * np.sqrt(aptarg.area) #sigma of bkg flux of target aperture

            error = calc_total_error(data, rms_map, 1.0) 
            target_photometry = aperture_photometry(data, aptarg, error=error)
            target_photometry['aperture_sum_bkgsub'] = target_photometry['aperture_sum'] - target_bkg #background subtracted total flux of target

            # Save the data
            self.results.add_row([
                0,
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
                None,
                None,
                None,
                None
            ])
        
        self.sort_results()
        return