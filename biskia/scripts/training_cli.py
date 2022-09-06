'''
Created on 24.08.2020

@author: Philipp
'''
import click

from biskia.configuration import ExperimentConfigurations
from biskia.training import perform_training_series_or_single


@click.command()
@click.argument("experiment_name")
@click.option("-c", "--config_directory_path", required=True,
              type=click.Path(exists=True),
              help="Path to the top directory holding the configuration files")
@click.option("-m", "--checkpoint_dir", required=True,
              type=click.Path(exists=True),
              help="Absolute path to a directory for loading and saving models")
@click.option("-d", "--dataset_dir", required=True,
              type=click.Path(exists=True),
              help="Path to the top directory holding the dataset e.g. /data/blockworld")
@click.option("-r", "--resume", required=False, type=bool, help="Load the checkpoint and continue training from there.")
@click.option("-u", "--comet_user", required=False, help="Name of the CometML config to use.")
@click.option("--log_grads", required=False, type=bool, default=False,
              help="Enable gradient logging. This might slow down the training. Thus only use for debugging.")
def training(experiment_name, config_directory_path, checkpoint_dir, dataset_dir, comet_user, resume,
             log_grads):
    """
        Perform the training for the given EXPERIMENT_NAME.

        There must be a sub-directory with name 'train' and one with 'dev' in the dataset directory configured in
        the dataset configuration. These directories are supposed to contain the training data.
        Use like:
        $> biskia-training experiment \
            -c /projects/biskia/configs -m /data/checkpoints/biskia -d /data/blockworld -u phisad

        EXPERIMENT_NAME The name of the experiment config to execute. The file has to be found in the 'experiments'
                        directory of the top config directory. The .json file-ending is automatically added.
        """
    configs = ExperimentConfigurations(config_directory_path, checkpoint_dir, dataset_dir, comet_user)
    config = configs.get_experiment_by_name(experiment_name, resume, log_grads)
    perform_training_series_or_single(config)
