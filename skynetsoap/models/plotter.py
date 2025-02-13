from astropy.time import Time, TimeDelta
import os
import matplotlib.pyplot as plt
import numpy as np

# DONE: Plot by filter
# TODO: Time parameters
# TODO: Create a plot for each filter if specified

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

filter_colors = {
    "Open": "k",

    "uprime": "tab:blue",
    "gprime": "tab:green",
    "rprime": "tab:red",
    "iprime": "tab:orange",
    "zprime": "tab:purple"
}

class Plotter:
    def __init__(self, results, units="flux"):
        if not os.path.exists("soap_results"):
            os.makedirs("soap_results")

        self.results = results
        self.units = units
        self.unit_err = self.units + "_err"


    def create_plot(self, path="soap_results/", multi_band=True, after=None, before=None, days_ago=None):
        """Create a plot from the photometry results."""
        self.results = self.filter_by_date(after, before, days_ago)
        if multi_band:                
            fig, ax = plt.subplots()
            for result in self.results:
                ax.errorbar(result["mjd"], result[self.units], yerr=result[self.unit_err], fmt='o', color=filter_colors.get(result['filter'], 'g'))
            ax.set_title(title[self.units])
            ax.set_xlabel("MJD")
            ax.set_ylabel(ylabel[self.units])
            filters = np.unique(self.results['filter'])
            ax.legend(filters)
            plt.savefig(f"{path}{self.units}_plot.png")
        else:
            filters = np.unique(self.results['filter'])
            for filter in filters:
                fig, ax = plt.subplots()
                filter_results = self.results[self.results["filter"] == filter]
                ax.errorbar(filter_results["mjd"], filter_results[self.units], yerr=filter_results[self.unit_err], fmt='o', color=filter_colors.get(filter, 'g'))
                ax.set_title(title[self.units])
                ax.set_xlabel("MJD")
                ax.set_ylabel(ylabel[self.units])
                ax.legend([filter])

                plt.savefig(f"{path}{self.units}_{filter}_plot.png")
        return
    

    def filter_by_date(self, after=None, before=None, days_ago=None):
        """Filter results by date."""
        if days_ago is not None:
            if after is not None or before is not None:
                raise ValueError("Please specify either 'days_ago' or 'after'/'before'.")
            after = Time.now(format='mjd') - TimeDelta(days_ago, format='jd')

        if after is not None:
            after = Time(after).mjd
            self.results = self.results[self.results["mjd"] > after]
        if before is not None:
            before = Time(before).mjd
            self.results = self.results[self.results["mjd"] < before]
        return self.results
