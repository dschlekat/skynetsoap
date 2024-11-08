import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skynetsoap.utils.skynet_api import SkynetAPI

obs_id = 11794606
obs = SkynetAPI().get(obs_id)

print(obs.name)