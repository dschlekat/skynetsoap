from skynetsoap import Soap

obs_id = 11920700
obs_name = 'cs_1246'
obj_ra = '12:49:37.598' 
obj_dec = '-63:32:09.8'

try:
    soap = Soap(obj_ra, obj_dec, obs_id, obs_name, calibrate=False)
    soap.download_images()
    soap.photometry_pipeline()
    soap.generate_plot(units='normalized_flux')
    soap.generate_table(filetype='astropy')
except KeyboardInterrupt:
    print('Exiting...')