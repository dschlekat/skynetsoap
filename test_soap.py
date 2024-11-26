from skynetsoap import Soap

obs_id = 9686357
obs_name = 'lyra obs 0'
obj_ra = '18:42:28.81'
obj_dec = '47:37:12.51'

soap = Soap(obj_ra, obj_dec, obs_id, calibrate=False)