from astropy.coordinates import SkyCoord
from skynetsoap import Soap

# JC filter calibration example
print("Running JC filter calibration example...")
s = Soap(observation_id=13110404, verbose=True)
s.download()
result = s.run()

if len(result) > 0:
    s.export()
    print(f"Total measurements: {len(result)}")
    print(f"5σ limiting mag: {result.table['limiting_mag'][0]:.2f} mag")

    target = SkyCoord(
        "03:27:48.9394738008 +74:39:52.531563600", unit=("hourangle", "deg")
    )
    target_result = result.extract_target(target, radius_arcsec=3.0)
    if len(target_result) > 0:
        target_result.export()
        print(f"Target detections: {len(target_result)}")
    else:
        print("No target detections found")
else:
    print("No measurements in result")

# SDSS filter calibration example
print("\nRunning SDSS filter calibration example...")
s = Soap(observation_id=12313253, verbose=True)
s.download()
result = s.run()

if len(result) > 0:
    s.export()
    print(f"Total measurements: {len(result)}")
    print(f"5σ limiting mag: {result.table['limiting_mag'][0]:.2f} mag")
else:
    print("No measurements in result")
