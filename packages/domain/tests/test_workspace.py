"""
Workspace integrity and module import tests for Milestone 1.
"""
import unittest

class TestMilestone1Workspace(unittest.TestCase):
    def test_domain_module(self):
        import domain
        self.assertEqual(domain.__version__, "0.1.0")

    def test_contracts_module(self):
        import contracts
        self.assertEqual(contracts.__version__, "0.1.0")

    def test_config_module(self):
        import config
        self.assertEqual(config.__version__, "0.1.0")

    def test_api_module(self):
        import api
        self.assertEqual(api.__version__, "0.1.0")

    def test_ingestion_module(self):
        import ingestion
        self.assertEqual(ingestion.__version__, "0.1.0")

if __name__ == "__main__":
    unittest.main()
