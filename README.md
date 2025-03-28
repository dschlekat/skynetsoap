# soap
_Note that SOAP is a work in progress, and is not in its final intended state._

Skynet Single-Object Aperture Photometry (SOAP) allows users to easily conduct photometry on a single object with defined coordinates using the observation ID of a Skynet observation. Once photometry is complete, SOAP can create photometric tables and lightcurves.

The core functionality of the package resides within the soap class, which is initialized with the observation ID and the coordinates of the object. The user can then call the photometry method to conduct the photometry, and the plot method to create lightcurves in a variety of units (flux, relative flux, magnitude). The user can also call the table method to create a photometric table or a GCN formatted table.

SOAP is designed to run using Python 3.12. The required packages are stored in the requirements.txt file, with the exception of the skynetapi package, which must be installed from the source code available for download at [https://github.com/astrodyl/skynetapi](https://github.com/astrodyl/skynetapi). To install the required packages (with pip installed), run the following command in the terminal: ```pip install -r requirements.txt```.
