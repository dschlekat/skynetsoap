import os
from skynetapi import ObservationRequest

class SkynetAPI:
    def __init__(self):
        self.api_key = os.getenv("SKYNET_API_KEY")
        if not self.api_key:
            raise ValueError("API key for Skynet is missing. Set it as an environment variable 'SKYNET_API_KEY'.")
        self.observation_request = ObservationRequest(api_key=self.api_key)

    def get(self, observation_id):
        return self.observation_request.get(observation_id)