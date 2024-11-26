import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from skynetsoap.models.observation import Observation

obs_id = 9686357
obs_name = 'lyra obs 0'
num_exps = 4

class TestObservation(unittest.TestCase):
    def setUp(self):
        self.observation = Observation(obs_id)

    def tearDown(self):
        if os.path.exists("soap_images"):
            shutil.rmtree("soap_images")

    def test_get_obs(self):
        name = self.observation.get_obs()
        self.assertEqual(name, obs_name)

    def test_download_images(self):
        obs = self.observation.get_obs()
        name = self.observation.download_images()
        self.assertEqual(name, obs_name)
        self.assertEqual(name, obs_name)
        self.assertTrue(os.path.exists("soap_images"))
        self.assertEqual(len(os.listdir("soap_images")), num_exps)


if __name__ == "__main__":
    unittest.main()