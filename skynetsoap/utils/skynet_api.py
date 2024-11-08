import os
from skynetapi import ObservationRequest

class SkynetAPI:
    def __init__(self):
        self.api_token = os.getenv("SKYNET_API_TOKEN")
        if not self.api_token:
            raise ValueError("API key for Skynet is missing. Set it as an environment variable 'SKYNET_API_TOKEN'.")
        self.observation_request = ObservationRequest(token=self.api_token)

    def get(self, observation_id):
        return self.observation_request.get(observation_id)