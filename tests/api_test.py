import os
from skynetapi import ObservationRequest

api_key = (os.getenv("SKYNET_API_KEY"))
if not api_key:
    raise ValueError("API key for Skynet is missing. Set it as an environment variable 'SKYNET_API_KEY'.")
obs = ObservationRequest(token=api_key).get(obs_id=11794606)

print(obs.name)