'''
Created on 24.08.2020

@author: Philipp
'''
import click

from biskia.configuration import ExperimentConfigurations
# from biskia.prediction import perform_prediction_series_or_single
from biskia.paraphrase import perform_paraphrase_single

@click.command()
@click.argument("experiment_name")
@click.argument("split_name")
@click.option("-c", "--config_directory_path", required=True,
              type=click.Path(exists=True),
              help="Path to the top directory holding the configuration files")
@click.option("-m", "--checkpoint_dir", required=True,
              type=click.Path(exists=True),
              help="Absolute path to a directory for loading the models")
@click.option("-d", "--dataset_dir", required=True,
              type=click.Path(exists=True),
              help="Path to the top directory holding the dataset e.g. /data/blockworld")
@click.option("-f", "--dataset_file_path", required=False,
              type=click.Path(exists=True),
              help="Path to a custom dataset file for prediction e.g. /usr/data.json. Use 'file' as split name.")
@click.option("-u", "--comet_user", required=False, help="Name of the CometML config to use.")
def paraphrase(experiment_name, split_name, config_directory_path, checkpoint_dir, dataset_dir, dataset_file_path,
            comet_user):
    # biskib-training rnn-predict-semantics -c configs -m /Users/lukaborec/Projects/IM/checkpoints -d /Users/lukaborec/Projects/IM/060_bisk_interpreter_data/data/blockworld -f /Users/lukaborec/Projects/IM/060_bisk_interpreter_data/data/blockworld/MNIST/semantics/Instructions.json -u lukaborec
    """
        Perform a prediction on the SPLIT_NAME for the given EXPERIMENT_NAME. The checkpoint is automatically
        loaded from the 'checkpoint_dir' configuration attribute.

        Use like:
        $> biskia-predict experiment split \
            -c /projects/biskia/configuration -m /data/checkpoints/biskia -d /data/blockworld -u phisad

        EXPERIMENT_NAME The name of the experiment config to execute. The file has to be found in the 'experiments'
                        directory of the top config directory. Must include the .json file-ending.
        SPLIT_NAME      Name of the dataset split on which to perform the prediction. Must be a sub-directory name
                        of the dataset directory configured in the dataset configuration. If the name is 'file'
                        then there must be also a path to a file given by using the '-f' option.
        """
    print("Entering script")
    if split_name == "file":
        if dataset_file_path is None:
            raise Exception("A dataset_file_path must be given when split-name is file.")
    else:
        if dataset_file_path is not None:
            print("The dataset_file_path will be ignored, because split-name is not 'file'.")
    configs = ExperimentConfigurations(config_directory_path, checkpoint_dir, dataset_dir, comet_user)
    config = configs.get_experiment_by_name(experiment_name, dataset_file_path=dataset_file_path)
    perform_paraphrase_single(config, split_name)
