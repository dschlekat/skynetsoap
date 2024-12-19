import unittest
import sys
import os
import astropy.table as tbl
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from skynetsoap.models.table import Table

class TestTable(unittest.TestCase):
    def setUp(self):
        self.results = [
            {'mjd': 59000.0, 'telescope': 'RRRT', 'filter': 'V', 'exp_len': 70, 'magnitude': 15.227, 'magnitude_err': 0.024, 'flux': 1.23, 'flux_err': 0.01, 'normalized_flux': 1.23, 'normalized_flux_err': 0.01},
            {'mjd': 59001.0, 'telescope': 'RRRT', 'filter': 'V', 'exp_len': 70, 'magnitude': 15.327, 'magnitude_err': 0.025, 'flux': 1.45, 'flux_err': 0.02, 'normalized_flux': 1.23, 'normalized_flux_err': 0.01},
            {'mjd': 59002.0, 'telescope': 'RRRT', 'filter': 'R', 'exp_len': 80, 'magnitude': 15.100, 'magnitude_err': 0.020, 'flux': 1.67, 'flux_err': 0.03, 'normalized_flux': 1.23, 'normalized_flux_err': 0.01},
        ]
        self.results = tbl.Table(self.results)
        self.table = Table(self.results)
        self.maxDiff = None

    def tearDown(self):
        if os.path.exists("soap_results/photometry_table.csv"):
            os.remove("soap_results/photometry_table.csv")
        if os.path.exists("soap_results/photometry_table.txt"):
            os.remove("soap_results/photometry_table.txt")
        if os.path.exists("soap_results/gcn_table.txt"):
            os.remove("soap_results/gcn_table.txt")
        if os.path.exists("soap_results/gcn_table_start.txt"):
            os.remove("soap_results/gcn_table_start.txt")
        if os.path.exists("soap_results/gcn_table_all.txt"):
            os.remove("soap_results/gcn_table_all.txt")

    def test_create_csv_table(self):
        self.table.create_table("csv")
        self.assertTrue(os.path.exists("soap_results/photometry_table.csv"))
        with open("soap_results/photometry_table.csv", "r") as f:
            content = f.read()
        expected_content = "mjd,flux,error\n59000.0,1.23,0.01\n59001.0,1.45,0.02\n59002.0,1.67,0.03\n"
        self.assertEqual(content, expected_content)

    def test_create_txt_table(self):
        self.table.create_table("txt")
        self.assertTrue(os.path.exists("soap_results/photometry_table.txt"))
        with open("soap_results/photometry_table.txt", "r") as f:
            content = f.read()
        expected_content = "mjd\tflux\terror\n59000.0\t1.23\t0.01\n59001.0\t1.45\t0.02\n59002.0\t1.67\t0.03\n"
        self.assertEqual(content, expected_content)

    def test_create_gcn_table_without_start_time(self):
        self.table.create_gcn_table(start_time=None, all_results=False)
        self.assertTrue(os.path.exists("soap_results/gcn_table.txt"))

        with open("soap_results/gcn_table.txt", "r") as f:
            content = f.read()
        
        # Expected content when start_time=None (MJD column)
        expected_content = (
            "MJD             | Telescope | Filter | Exposure Duration | Mag     | Mag Error\n"
            "59000.00000 MJD | RRRT      | V      | 70s               | 15.227  | 0.024    \n"    
            "59002.00000 MJD | RRRT      | R      | 80s               | 15.100  | 0.020    \n"
        )
        self.assertEqual(content, expected_content)

    def test_create_gcn_table_with_start_time(self):
        start_time = 59000.0
        self.table.create_gcn_table(start_time=start_time, all_results=False)
        self.assertTrue(os.path.exists("soap_results/gcn_table_start.txt"))

        with open("soap_results/gcn_table_start.txt", "r") as f:
            content = f.read()
        
        # Expected content when start_time is provided
        expected_content = (
            "Time Since GRB | Telescope | Filter | Exposure Duration | Mag     | Mag Error\n"
            "0s             | RRRT      | V      | 70s               | 15.227  | 0.024    \n"
            "172800s        | RRRT      | R      | 80s               | 15.100  | 0.020    \n"
        )
        self.assertEqual(content, expected_content)

    def test_create_gcn_table_all_results(self):
        # Test including all results (all_results=True)
        self.table.create_gcn_table(start_time=None, all_results=True)
        self.assertTrue(os.path.exists("soap_results/gcn_table_all.txt"))

        with open("soap_results/gcn_table_all.txt", "r") as f:
            content = f.read()
        
        # Expected content including all results for each filter
        expected_content = (
            "MJD             | Telescope | Filter | Exposure Duration | Mag     | Mag Error\n"
            "59000.00000 MJD | RRRT      | V      | 70s               | 15.227  | 0.024    \n"
            "59001.00000 MJD | RRRT      | V      | 70s               | 15.327  | 0.025    \n"
            "59002.00000 MJD | RRRT      | R      | 80s               | 15.100  | 0.020    \n"
        )
        self.assertEqual(content, expected_content)

    def test_invalid_filetype(self):
        with self.assertRaises(ValueError):
            self.table.create_table("invalid")

if __name__ == "__main__":
    unittest.main()