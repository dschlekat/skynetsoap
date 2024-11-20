import os
from skynetapi import ObservationRequest, DownloadRequest

class SkynetAPI:
    def __init__(self):
        if not os.path.exists("soap_images"):
            os.makedirs("soap_images")

        self.api_token = os.getenv("SKYNET_API_TOKEN")
        if not self.api_token:
            raise ValueError("API key for Skynet is missing. Set it as an environment variable 'SKYNET_API_TOKEN'.")
        self.observation_request = ObservationRequest(token=self.api_token)
        self.download_request = DownloadRequest(token=self.api_token)

    def get_observation(self, observation_id):
        return self.observation_request.get(observation_id)
    
    def download_all_images(self, observation, path="soap_images/"):
        filepaths = []
        for exp in observation.exps:
            filepaths += self.download_fits(exp.id, path=path)
        return filepaths

    def download_fits(self, observation_id, path="soap_images/"):
        filepath = self.download_request.get_fits(out_dir=path, reducequiet=1, **{'image': f'r{observation_id}'})
        return filepath