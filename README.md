# soap
_Note that this is a work in progress, and is not in a usable state._

SOAP (Skynet single-Object Aperture Photometry) allows users to easily conduct photometry on a single object with defined coordinates using the observation ID of the relevant Skynet observation. Once the photometry is complete, SOAP can create photometric tables and lightcurves.

The core functionality of the package resides within the soap class, which is initialized with the observation ID and the coordinates of the object. The user can then call the photometry method to conduct the photometry, and the plot method to create lightcurves in a variety of units (flux, relative flux, magnitude). The user can also call the table method to create a photometric table or a GCN formatted table. 
