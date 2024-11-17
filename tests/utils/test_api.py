import sys
import unittest
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from skynetsoap.utils.skynet_api import SkynetAPI

obs_id = 9686357
obs_name = 'lyra obs 0'
num_exps = 4

class TestSkynetAPI(unittest.TestCase):
    def setUp(self):
        self.api = SkynetAPI()

    def test_get(self):
        obs = self.api.get_observation(obs_id)
        self.assertEqual(obs.id, obs_id)
        self.assertEqual(obs.name, obs_name)
        self.assertEqual(len(obs.exps), num_exps)

if __name__ == "__main__":
    unittest.main()
