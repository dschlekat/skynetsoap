from skynetsoap import Soap

obs_id = 11920699
obs_name = 'cs_1246'
obj_ra = '12:49:37.598'
obj_dec = '-63:32:09.8'
after_date = '2025-01-12T00:00:00'

try:
    soap = Soap(obj_ra, obj_dec, obs_id, obs_name)
    soap.download_images(after=after_date)
    soap.photometry_pipeline()
    soap.generate_table(filetype='astropy', after=after_date)
    soap.generate_plot(units='normalized_flux', after=after_date)
except KeyboardInterrupt:
    print('Exiting...')