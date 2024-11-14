from skynetsoap.utils import SkynetAPI

class Observation:
    def __init__(self, observation_id):
        self.observation_id = observation_id
        self.obs = None
        self.images = None

    def get_obs(self):
        """Get the desired observation using the Skynet API."""
        self.obs = SkynetAPI().get(self.observation_id)
        return self.obs.name
    
    def get_image(image_id):
        """Get the desired image using the Skynet API."""
        pass
    
    def download_images(self):
        """Download the images for the desired observation."""
        pass