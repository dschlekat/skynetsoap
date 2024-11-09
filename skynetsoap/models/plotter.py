import os
import matplotlib.pyplot as plt

title = {
    "flux": "Flux vs Time",
    "normalized_flux": "Normalized Flux vs Time",
    "magnitude": "Magnitude vs Time"
}

ylabel = {
    "flux": "Flux",
    "normalized_flux": "Normalized Flux",
    "magnitude": "Magnitude"
}

class Plotter:
    def __init__(self, results, units="flux"):
        self.results = results
        self.units = units

    def create_plot(self, path="soap_results/photometry_plot"):
        if not os.path.exists("soap_results"):
            os.makedirs("soap_results")
        
        
        fig, ax = plt.subplots()
        ax.errorbar([result["mjd"] for result in self.results], 
                    [result[self.units] for result in self.results], 
                    yerr=[result["error"] for result in self.results], 
                    fmt='o')
        ax.set_title(title[self.units])
        ax.set_xlabel("MJD")
        ax.set_ylabel(ylabel[self.units])
        plt.savefig(f"{path}.png") 
        return
