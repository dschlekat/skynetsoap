from photutils.utils import ImageDepth

class LimitingMagnitude:
    def __init__(self, image, aperture_radius):
        self.image = image
        self.aperture_radius = aperture_radius
        self.depth = None

    def calculate_depth(self):
        """Calculate the image depth."""
        self.depth = ImageDepth(self.aperture_radius)
        return self.depth