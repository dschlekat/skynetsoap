from skynetsoap.utils import SkynetAPI

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
    
    def download_images(self):
        """Download the images for the desired observation."""
        check = self.check_for_images()
        self.images = self.api.download_all_images(self.obs)
        return self.obs.name
    
    def check_for_images(self):
        """Check for exsiting images"""
        