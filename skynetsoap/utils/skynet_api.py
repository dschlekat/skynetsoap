import os
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
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
    
    def download_images(self, observation, path="soap_images/", after=None, before=None, days_ago=None):
        """
        Downloads FITS images for exposures within a given datetime range.

        :param observation: The observation object containing exposures.
        :param path: The directory where images will be saved.
        :param after: Only download images from exposures taken after this datetime.
        :param before: Only download images from exposures taken before this datetime.
        :param days_ago: If set, only download images from this many days ago until now.
        :return: A list of file paths of the downloaded images.
        """
        count = 0
        if days_ago is not None:
            after = datetime.now(timezone.utc) - timedelta(days=days_ago)
        if after is not None:
            after = datetime.fromisoformat(after)
        if before is not None:
            before = datetime.fromisoformat(before)

        loop = tqdm(observation.exps)
        for exp in loop:
            loop.set_description(f"Downloading {exp.id}")
            try:
                if exp.center_time is None:
                    loop.set_description(f"Skipping exposure {exp.id}: Not taken yet.")
                    continue
                center_time = datetime.fromisoformat(exp.center_time)  # Convert string to datetime
            except TypeError:
                loop.set_description("Skipping exposure {exp.id}: Invalid datetime format '{exp.center_time}'")
                continue

            if after is not None and center_time < after:
                continue
            if before is not None and center_time > before:
                continue
            self.download_fits(exp.id, path=path)
            count += 1

        print(f"Downloaded {count} images.")

    def download_fits(self, observation_id, path="soap_images/"):
        filepath = self.download_request.get_fits(out_dir=path, reducequiet=1, **{'image': f'r{observation_id}'})
        return filepath