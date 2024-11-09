import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skynetsoap.models.table import Table

class TestTable(unittest.TestCase):
    def setUp(self):
        self.results = [
            {'mjd': 59000.0, 'flux': 1.23, 'error': 0.01},
            {'mjd': 59001.0, 'flux': 1.45, 'error': 0.02},
            {'mjd': 59002.0, 'flux': 1.67, 'error': 0.03}
        ]
        self.table = Table(self.results)

    def tearDown(self):
        if os.path.exists("soap_results/photometry_table.csv"):
            os.remove("soap_results/photometry_table.csv")
        if os.path.exists("soap_results/photometry_table.txt"):
            os.remove("soap_results/photometry_table.txt")

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
        expected_content = "mjd,flux,error\n59000.0\t1.23\t0.01\n59001.0\t1.45\t0.02\n59002.0\t1.67\t0.03\n"
        self.assertEqual(content, expected_content)

    def test_invalid_filetype(self):
        with self.assertRaises(ValueError):
            self.table.create_table("invalid")

if __name__ == "__main__":
    unittest.main()