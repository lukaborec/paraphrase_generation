"""
Basic logging configuration for the whole project
"""
import logging
import os
import tempfile

from comet_ml import Experiment, OfflineExperiment
import torch
from torch import cuda
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__file__)


def __load_device(force_cpu=False):
    if force_cpu:
        device_name = "cpu"
    else:
        device_name = "cuda:0" if cuda.is_available() else "cpu"
    return torch.device(device_name)


def __load_loss_fn(loss_config):
    return __load_loss_fn_dynamically(loss_config["package"], loss_config["class"])


def __load_loss_fn_dynamically(python_package, python_class):
    loss_package = __import__(python_package, fromlist=python_class)
    loss_class = getattr(loss_package, python_class)
    return loss_class()


def __load_model_from_config(model_config, task):
    return __load_model_dynamically(model_config["package"], model_config["class"], model_config["params"], task)


def __load_model_dynamically(python_package, python_class, model_params, task):
    model_package = __import__(python_package, fromlist=python_class)
    model_class = getattr(model_package, python_class)
    return model_class(task, model_params)


def __load_dataset_from_config(dataset_config, task, split_name, device):
    return __load_dataset_dynamically(dataset_config["package"], dataset_config["class"], dataset_config["params"],
                                      task, split_name, device)


def __load_dataset_dynamically(python_package, python_class, dataset_params, task, split_name, device):
    dataset_package = __import__(python_package, fromlist=python_class)
    dataset_class = getattr(dataset_package, python_class)
    return dataset_class(task, split_name, dataset_params, device)


def __collate_variable_sequences(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    return x, y  # returning a pair of tensor-lists


def __load_cometml_experiment(comet_config, experiment_name, tags):
    # Optionals
    cometml_workspace = None
    cometml_project = None
    if "workspace" in comet_config:
        cometml_workspace = comet_config["workspace"]
    if "project_name" in comet_config:
        cometml_project = comet_config["project_name"]

    if comet_config["offline"]:
        # Optional offline directory
        offline_directory = None
        if "offline_directory" in comet_config:
            offline_directory = comet_config["offline_directory"]
        # Defaults to tmp-dir
        if offline_directory is None:
            offline_directory = os.path.join(tempfile.gettempdir(), "cometml")
        if not os.path.exists(offline_directory):
            os.makedirs(offline_directory)
        logger.info("Writing CometML experiments to %s", offline_directory)
        experiment = OfflineExperiment(workspace=cometml_workspace, project_name=cometml_project,
                                       offline_directory=offline_directory)
    else:
        experiment = Experiment(workspace=cometml_workspace, project_name=cometml_project,
                                api_key=comet_config["api_key"])
    experiment.set_name(experiment_name)
    experiment.add_tags(tags)
    return experiment


def __log_parameters(experiment, experiment_config):
    def apply_prefix(params_dict, prefix):
        return dict([("%s-%s" % (prefix, name), v) for name, v in params_dict.items()])

    experiment.log_parameters(apply_prefix(experiment_config["params"], "exp"))
    experiment.log_parameters(apply_prefix(experiment_config["model"]["params"], "model"))
    experiment.log_parameters(apply_prefix(experiment_config["dataset"]["params"], "ds"))
    experiment.log_parameters(apply_prefix(experiment_config["task"], "task"))


class Saver(object):

    def __init__(self, checkpoint_top_dir, experiment_name, model_name, metric, mode="highest"):
        self.checkpoint_dir = os.path.join(checkpoint_top_dir, experiment_name)
        self.model_name = model_name
        self.metric = metric
        if mode == "highest":
            self.comparator = lambda x, y: x > y
            self.best_value = 0
            self.comparator_string = ">"
        if mode == "lowest":
            import math
            self.comparator = lambda x, y: x < y
            self.best_value = math.inf
            self.comparator_string = "<"

    def save_checkpoint_if_best(self, model, optimizer, epoch, metrics):
        epoch_value = metrics[self.metric].to_value()
        if self.comparator(epoch_value, self.best_value):
            print("Save checkpoint at epoch %s: epoch_value %s best_value (%.3f %s %.3f) [%s]" %
                  (str(epoch), self.comparator_string, epoch_value, self.comparator_string, self.best_value,
                   self.checkpoint_dir))
            self.best_value = epoch_value
            self.save_checkpoint(model, optimizer, epoch)

    def save_checkpoint(self, model, optimizer, epoch):
        if not os.path.exists(self.checkpoint_dir):
            # logger.info("Create experiment checkpoint directory at %s", experiment_checkpoint_dir)
            os.makedirs(self.checkpoint_dir)
        torch.save({
            'epoch': epoch,
            'arch': self.model_name,
            'state_dict': model.state_dict(),
            'best_value': self.best_value,
            'best_value_metric': self.metric,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, "model_best.pth.tar"))

    @staticmethod
    def load_checkpoint(model, experiment, checkpoint_dir, experiment_name):
        experiment_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        experiment_checkpoint_path = os.path.join(experiment_checkpoint_dir, "model_best.pth.tar")
        if not os.path.exists(experiment_checkpoint_path):
            raise Exception("Cannot find experiment checkpoint at %s", experiment_checkpoint_path)
        checkpoint = torch.load(experiment_checkpoint_path)
        if experiment:
            experiment.log_other("checkpoint_epoch", checkpoint["epoch"])
            experiment.log_other("checkpoint_best_value", checkpoint["best_value"])
            experiment.log_other("checkpoint_best_value_metric", checkpoint["best_value_metric"])
            experiment.log_other("checkpoint_arch", checkpoint["arch"])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return checkpoint


def __load_interpreter(interpreter_config, interpreter_name, experiment_config, device):
    """ Load the interpreter for evaluation """
    interpreter = __load_model_from_config(interpreter_config, experiment_config["task"])
    Saver.load_checkpoint(interpreter, None, experiment_config["checkpoint_dir"], interpreter_name)
    interpreter.to(device)
    interpreter.requires_grad_(False)
    return interpreter
