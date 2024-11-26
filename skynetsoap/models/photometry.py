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
import tqdm

# stop Astropy warnings
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

class Photometry:
    def __init__(self, ra, dec, images):
        # if ra is in hms, convert to degrees
        if isinstance(ra, str):
            ra = np.degrees(float(ra.split()[0]) + float(ra.split()[1])/60 + float(ra.split()[2])/3600)
        if isinstance(dec, str):
            dec = np.degrees(float(dec.split()[0]) + float(dec.split()[1])/60 + float(dec.split()[2])/3600)
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
        ))


    def open_fits(self, image):
        """Open an image using astropy."""
        hdulist = fits.open(image)
        data = hdulist[0].data
        header = hdulist[0].header
        hdulist.close()
        return data, header
    
    def calibrate_magnitudes(self):
        """Calibrate magnitudes using a catalog."""
        return
    
    def run_pipeline(self):
        """Run aperture photometry on the downloaded images."""
        loop = tqdm.tqdm(self.images)
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
                xtarg, 
                ytarg, 
                target_photometry['aperture_sum_bkgsub'][0], 
                error[0], 
                None, 
                None, 
                None, 
                None
            ])

        self.normalize_flux()
        return
    
    def normalize_flux(self):
        # Create a lightcurve using the table data
        cleaned_table = self.results[~np.isnan(self.results['flux'])]
        mjd_first = np.min(cleaned_table['mjd'].data)

        flux = cleaned_table['flux'].data
        flux_err = cleaned_table['flux error'].data
        mjd = cleaned_table['mjd'].data

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

    
    
            