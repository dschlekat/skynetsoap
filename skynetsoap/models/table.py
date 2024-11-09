import os

class Table:
    def __init__(self, results, units="flux"):
        self.results = results
        self.units = units
        

    def create_table(self, filetype, path="soap_results/photometry_table"):
        if not os.path.exists("soap_results"):
            os.makedirs("soap_results")
        
        if filetype == "csv":
            return self.create_csv_table(path)
        elif filetype == "txt":
            return self.create_txt_table(path)
        else:
            raise ValueError("Invalid filetype. Please specify 'csv' or 'txt'.")
        
    def create_csv_table(self, path):
        """Create a CSV table from the photometry results."""
        table = f"mjd,{self.units},error\n"
        for result in self.results:
            table += f"{result['mjd']},{result[self.units]},{result['error']}\n"
        filepath = f"{path}.csv"
        with open(filepath, "w") as f:
            f.write(table)
        return 
    
    def create_txt_table(self, path):
        """Create a text table from the photometry results."""
        table = f"mjd,{self.units},error\n"
        for result in self.results:
            table += f"{result['mjd']}\t{result[self.units]}\t{result['error']}\n"
        filepath = f"{path}.txt"
        with open(filepath, "w") as f:
            f.write(table)
        return 