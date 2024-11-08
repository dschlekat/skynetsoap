from skynetsoap.utils import SkynetAPI

class Observation:
    def __init__(self, observation_id):
        self.observation_id = observation_id
        self.images = None

    def download_images(self):
        """Download images using the Skynet API."""
        self.images = SkynetAPI().get(self.observation_id)