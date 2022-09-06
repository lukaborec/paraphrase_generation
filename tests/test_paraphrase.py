"""
Created on 24.08.2020

@author: Philipp
"""
import unittest
from biskia.configuration import ExperimentConfigurations
from biskia.paraphrase import perform_paraphrase_single
from biskia.prediction import perform_prediction_series_or_single


class Test(unittest.TestCase):
    config_directory_path = "../configs"
    checkpoint_dir = "F:/Development/data/checkpoints/biskia"
    dataset_dir = "F:/Development/data/blockworld"
    file_path = "F:/Development/data/blockworld/MNIST/semantics/Custom.json"
    user = "phisad"

    def test_lstm_from_semantics_file(self):
        configs = ExperimentConfigurations(Test.config_directory_path, Test.checkpoint_dir, Test.dataset_dir, Test.user)
        config = configs.get_experiment_by_name("lstm-from-semantics-init-step", dataset_file_path=Test.file_path)
        perform_paraphrase_single(config, split_name="file")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.test_training']
    unittest.main()
