from astropy.coordinates import SkyCoord
from skynetsoap import Soap

# JC filter calibration example
print("Running JC filter calibration example...")
s = Soap(observation_id=13110404, verbose=True)
s.download()
result = s.run()
s.export()

target = SkyCoord("03:27:48.9394738008 +74:39:52.531563600", unit=("hourangle", "deg"))
target_result = result.extract_target(target, radius_arcsec=3.0)
target_result.export()

# SDSS filter calibration example
print("Running SDSS filter calibration example...")
s = Soap(observation_id=12313253, verbose=True)
s.download()
result = s.run()
s.export()
