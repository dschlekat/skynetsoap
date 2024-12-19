import os
from skynetsoap.utils import SkynetAPI

# TODO: Implement a better check for existing images, can be done with expected # of fits files within the dir

class Observation:
    def __init__(self, observation_id):
        self.observation_id = observation_id
        self.obs = None
        self.images = None

        self.api = SkynetAPI()

    def get_obs(self):
        """Get the desired observation using the Skynet API."""
        self.obs = self.api.get_observation(self.observation_id)
        return self.obs.name
    
    def download_images(self, path="soap_images/"):
        """Download the images for the desired observation."""
        check = self.check_for_images(path=path)
        if check: 
            print('Images already downloaded')
            self.images = os.listdir(path)
            return self.obs.name
        else:
            self.images = self.api.download_all_images(self.obs, path=path)
            return self.obs.name
    
    def check_for_images(self, path):
        """Check for exsiting images"""
        if os.path.exists(path):
            if len(os.listdir(path)) > 0:
                return True
            else:
                return False
        else:
            return False


        