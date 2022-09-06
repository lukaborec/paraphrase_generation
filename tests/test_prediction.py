"""
Created on 24.08.2020

@author: Philipp
"""
import unittest
from biskia.configuration import ExperimentConfigurations
from biskia.prediction import perform_prediction_series_or_single


class Test(unittest.TestCase):
    config_directory_path = "../configs"
    checkpoint_dir = "F:/Development/data/checkpoints/biskia"
    dataset_dir = "F:/Development/data/blockworld"
    file_path = "F:/Development/data/blockworld/MNIST/semantics/Custom.json"
    user = "phisad"

    def test_lstm_from_locations_file(self):
        configs = ExperimentConfigurations(Test.config_directory_path, Test.checkpoint_dir, Test.dataset_dir, Test.user)
        config = configs.get_experiment_by_name("lstm-from-locations-init-step")
        perform_prediction_series_or_single(config, split_name="dev")

    def test_lstm_from_semantics_file(self):
        configs = ExperimentConfigurations(Test.config_directory_path, Test.checkpoint_dir, Test.dataset_dir, Test.user)
        config = configs.get_experiment_by_name("lstm-from-semantics-init-step", dataset_file_path=Test.file_path)
        perform_prediction_series_or_single(config, split_name="file")

    def test_lstm_from_semantics(self):
        configs = ExperimentConfigurations(Test.config_directory_path, Test.checkpoint_dir, Test.dataset_dir)
        config = configs.get_experiment_by_name("lstm-from-semantics-init-step")
        perform_prediction_series_or_single(config, split_name="dev")

    def test_prediction_single_from_config(self):
        configs = ExperimentConfigurations(Test.config_directory_path, Test.checkpoint_dir, Test.dataset_dir)
        config = configs.get_experiment_by_name("lstm-attentional")
        perform_prediction_series_or_single(config, split_name="dev")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_training']
    unittest.main()
