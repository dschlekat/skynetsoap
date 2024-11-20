import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from skynetsoap.utils.skynet_api import SkynetAPI

obs_id = 9686357
obs_name = 'lyra obs 0'
num_exps = 4

class TestSkynetAPI(unittest.TestCase):
    def setUp(self):
        self.api = SkynetAPI()

    def tearDown(self) -> None:
        if os.path.exists("soap_images"):
            shutil.rmtree("soap_images")

    def test_get(self):
        obs = self.api.get_observation(obs_id)
        self.assertEqual(obs.id, obs_id)
        self.assertEqual(obs.name, obs_name)
        self.assertEqual(len(obs.exps), num_exps)

    def test_download_fits(self):
        obs = self.api.get_observation(obs_id)
        fits = self.api.download_fits(obs.exps[0].id)
        self.assertTrue(os.path.exists(fits))

    def test_download_all_images(self):
        obs = self.api.get_observation(obs_id)
        filepaths = self.api.download_all_images(obs)
        num_files = len(os.listdir("soap_images"))
        self.assertEqual(num_files, num_exps)
    

if __name__ == "__main__":
    unittest.main()
