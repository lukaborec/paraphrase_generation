"""
Created on 24.08.2020

@author: Philipp
"""
import unittest
from biskia.configuration import ExperimentConfigurations
from biskia.training import perform_training_series_or_single


class Test(unittest.TestCase):
    config_directory_path = "../configs"
    checkpoint_dir = "F:/Development/data/checkpoints/biskia"
    dataset_dir = "F:/Development/data/blockworld"
    user = "phisad"

    def test_lstm_from_locations(self):
        configs = ExperimentConfigurations(Test.config_directory_path, Test.checkpoint_dir, Test.dataset_dir)
        config = configs.get_experiment_by_name("lstm-from-locations-init-step")
        perform_training_series_or_single(config)

    def test_lstm_from_semantics(self):
        configs = ExperimentConfigurations(Test.config_directory_path, Test.checkpoint_dir, Test.dataset_dir)
        config = configs.get_experiment_by_name("lstm-from-semantics-init-step")
        perform_training_series_or_single(config)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_training']
    unittest.main()
