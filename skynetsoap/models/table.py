from astropy.table import Table as tbl
from astropy.time import Time, TimeDelta
from .result import Result 
import os

class Table:
    def __init__(self, results, units="flux"):
        if not os.path.exists("soap_results"):
            os.makedirs("soap_results")

        self.results = results
        self.units = units
        self.unit_err = self.units + "_err"
        

    def create_table(self, filetype, path="soap_results/", after=None, before=None, days_ago=None):
        """Create a table from the photometry results given a filetype."""
        self.results = self.filter_by_date(after, before, days_ago)

        if filetype == "astropy":
            return self.create_astropy_table(path)
        elif filetype == "csv":
            return self.create_csv_table(path)
        elif filetype == "txt":
            return self.create_txt_table(path)
        elif filetype == "gcn":
            return self.create_gcn_table(path)
        else:
            raise ValueError("Invalid filetype. Please specify 'csv' or 'txt'.")
        
    def create_astropy_table(self, path):
        """Create an Astropy table from the photometry results."""
        table = tbl(self.results)
        filepath = f"{path}astropy_table.ecsv"
        table.write(filepath, format="ascii.ecsv", overwrite=True)
        return filepath
        
    def create_csv_table(self, path):
        """Create a CSV table from the photometry results."""
        table = f"mjd,{self.units},error\n"
        for result in self.results:
            table += f"{result['mjd']},{result[self.units]},{result[self.unit_err]}\n"
        filepath = f"{path}photometry_table.csv"
        with open(filepath, "w") as f:
            f.write(table)
        return 
    
    def create_txt_table(self, path):
        """Create a text table from the photometry results."""
        table = f"mjd\t{self.units}\terror\n"
        for result in self.results:
            table += f"{result['mjd']}\t{result[self.units]}\t{result[self.unit_err]}\n"
        filepath = f"{path}photometry_table.txt"
        with open(filepath, "w") as f:
            f.write(table)
        return 
    
    def create_gcn_table(self, path="soap_results/", start_time=None, all_results=False):
        """Create a GCN-formatted table from the photometry results."""
        
        # Adjust header based on start_time
        if start_time is None:
            header = "MJD             | Telescope | Filter | Exposure Duration | Mag     | Mag Error"
        else:
            header = "Time Since GRB | Telescope | Filter | Exposure Duration | Mag     | Mag Error"
        
        # Prepare data based on parameters
        rows = []
        unique_filters = set()
        
        for row in self.results:  # Astropy Table allows direct row iteration
            # Determine time column based on start_time
            if start_time is None:
                time_value = row["mjd"]
                time_str = f"{time_value:.5f} MJD"
            else:
                time_value = (row["mjd"] - start_time) * 86400  # Convert MJD difference to seconds
                time_str = f"{int(time_value)}s"
            
            # Collect data for the row
            telescope = row["telescope"]
            filter_name = row["filter"]
            exp_len = f"{int(row['exp_len'])}s"
            magnitude = f"{row['magnitude']:.3f}"
            mag_error = f"{row['magnitude_err']:.3f}"
            
            # Append first result per filter if all_results is False
            if not all_results:
                if filter_name in unique_filters:
                    continue
                unique_filters.add(filter_name)
            
            rows.append([time_str, telescope, filter_name, exp_len, magnitude, mag_error])
        
        # Write to file with formatted spacing
        name = "gcn_table"
        if start_time is not None:
            #append start to the file name
            name += "_start"
        if all_results:
            #append all to the file name
            name += "_all"
        filepath = f"{path}{name}.txt"
        
        with open(filepath, "w") as file:
            file.write(header + "\n")
            for row in rows:
                file.write("{:<14} | {:<9} | {:<6} | {:<17} | {:<7} | {:<9}\n".format(*row))

        print(f"GCN table saved to {filepath}")
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
    