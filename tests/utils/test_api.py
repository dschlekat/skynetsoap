import sys
import unittest
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from skynetsoap.utils.skynet_api import SkynetAPI

obs_id = 11794606
obs_name = 'SWIFT-BAT-1263718'

class TestSkynetAPI(unittest.TestCase):
    def setUp(self):
        self.api = SkynetAPI()

    def test_get(self):
        obs = self.api.get_observation(obs_id)
        self.assertEqual(obs.id, obs_id)
        self.assertEqual(obs.name, obs_name)

if __name__ == "__main__":
    unittest.main()
