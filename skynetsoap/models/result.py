from astropy.table import Table as tbl
from astropy.time import Time, TimeDelta

class Result:
    def __init__(self):
        self.results = tbl.Table(names=(
                'exp_number',
                'exp_name',            
                'telescope', 
                'filter', 
                'exp_len', 
                'mjd',
                'time_since_start',
                'xcenter', 
                'ycenter', 
                'flux', 
                'flux_err', 
                'normalized_flux', 
                'normalized_flux_err', 
                'magnitude', 
                'magnitude_err'
            ), dtype=(
                'i8',
                'U100', 
                'U100', 
                'U100', 
                'f8',
                'f8',
                'f8',
                'f8',
                'f8',
                'f8',
                'f8',
                'f8',
                'f8',
                'f8',
                'f8'
            ))
        
    def add_result(self, result):
        self.results.add_row(result)
        return
    
    def check_for_results(self):
        """Check if results already exist."""
        if len(self.results) > 0:
            return True
        else:  
            return False
    
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
        