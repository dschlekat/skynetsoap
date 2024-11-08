class Table:
    def __init__(self, results):
        self.results = results
        self.units = None
        self.normalize = None
        

    def create_table(self, filetype):
        if filetype == "csv":
            return self.create_csv_table()
        elif filetype == "txt":
            return self.create_txt_table()
        else:
            raise ValueError("Invalid filetype. Please specify 'csv' or 'txt'.")
        
    def create_csv_table(self):
        """Create a CSV table from the photometry results."""
        table = "mjd,flux,error\n"
        for result in self.results:
            table += f"{result['mjd']},{result['flux']},{result['error']}\n"
        return table
    
    def create_txt_table(self):
        """Create a text table from the photometry results."""
        table = "Time\tFlux\tError\n"
        for result in self.results:
            table += f"{result['mjd']}\t{result['flux']}\t{result['error']}\n"
        return table
    

        