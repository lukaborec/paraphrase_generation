'''
Created on 20.08.2020

@author: Philipp
'''
import json
import os

__EXPAND_KEYWORDS = ["model", "dataset", "task", "interpreter"]


def __expand_dict_values(config_top_directory_or_file, loaded_value):
    if not isinstance(loaded_value, dict):
        return loaded_value
    for key in loaded_value.keys():
        if key in __EXPAND_KEYWORDS and loaded_value[key].endswith(".json"):
            loaded_value[key] = load_json_config_as_dict(config_top_directory_or_file, loaded_value[key])
        else:  # go deeper if necessary
            loaded_value[key] = __expand_dict_values(config_top_directory_or_file, loaded_value[key])
    return loaded_value


def __expand_config_values(config_top_directory_or_file, loaded_config):
    config = dict()
    for key in loaded_config.keys():
        # these are special pointer keys to configs (which could also be inlined)
        if key in __EXPAND_KEYWORDS and loaded_config[key].endswith(".json"):
            config[key] = load_json_config_as_dict(config_top_directory_or_file, loaded_config[key])
        else:
            # if the value is a dict with values that refer to configs
            config[key] = __expand_dict_values(config_top_directory_or_file, loaded_config[key])
    if "series" in loaded_config.keys():  # special case that should only occur once in top level
        series = loaded_config["series"]
        config["series"] = [__expand_config_values(config_top_directory_or_file, entry) for entry in series]
        # copy all the values from the "series" config to each actual series entry
        for entry in config["series"]:
            for series_key in loaded_config.keys():
                if series_key not in ["name", "series", "params"]:  # do now overwrite name or copy the series
                    entry[series_key] = loaded_config[series_key]
                if series_key == "params":  # merge params if possible
                    if series_key in entry:
                        if isinstance(entry[series_key], dict) and isinstance(loaded_config[series_key], dict):
                            entry[series_key] = {**entry[series_key], **loaded_config[series_key]}
                    else:
                        entry[series_key] = loaded_config[series_key]
    return config


def load_json_config_as_dict(config_top_directory_or_file, relative_config_file_path=None):
    """
        :param config_top_directory_or_file:
        the top directory in which other config directories are located
        or an absolute path to a config file
        :param relative_config_file_path: the path to a config file relative to the config_top_directory_or_file.
        Can be None, when the other parameter is already pointing to a config file.
    """
    config_path = config_top_directory_or_file
    if os.path.isdir(config_top_directory_or_file):
        config_path = os.path.join(config_top_directory_or_file, relative_config_file_path)
    with open(config_path, "r", encoding="utf8", newline='') as json_file:
        loaded_config = json.load(json_file)
        expanded_config = __expand_config_values(config_top_directory_or_file, loaded_config)
    return expanded_config


def replace_placeholder(placeholder: str, value: str, parameters: dict):
    for name, parameter in parameters.items():
        if isinstance(parameter, dict):
            replace_placeholder(placeholder, value, parameter)
        if isinstance(parameter, str):
            parameters[name] = parameter.replace(placeholder, value)


class ExperimentConfigurations:

    def __init__(self, config_top_dir, checkpoint_dir, dataset_top_dir, comet_user=None, conditional=None):
        self.config_top_dir = config_top_dir
        self.experiment_configs = dict()
        self.conditional = conditional
        self.__load_experiments()

        # Load the cometml config
        if comet_user:
            cometml_config = load_json_config_as_dict(config_top_dir, "cometml/%s.json" % comet_user)
        else:
            cometml_config = load_json_config_as_dict(config_top_dir, "../configs/cometml/offline.json")

        # Prepare the configs for each experiment
        self.__prepare_configs(self.experiment_configs, checkpoint_dir, dataset_top_dir, cometml_config)

    def __prepare_configs(self, experiment_configs, checkpoint_dir, dataset_top_dir, cometml_config):
        for name, config in experiment_configs.items():
            if "series" in config:
                self.__prepare_configs(dict([(c["name"], c) for c in config["series"]]),
                                       checkpoint_dir,
                                       dataset_top_dir,
                                       cometml_config)
            else:
                config["cometml"] = cometml_config
                config["checkpoint_dir"] = checkpoint_dir
                config["dataset"]["params"]["dataset_directory"] = dataset_top_dir
                config["conditioanl"] = self.conditional
                # For aggregate models prefix the aggregates with checkpoint_dir (to achieve relative paths)
                replace_placeholder("$checkpoint_dir", checkpoint_dir, config["model"]["params"])

    def __load_experiments(self):
        """
            Experiments specify common parameters and combine models and datasets.
        """
        experiments_dir = os.path.join(self.config_top_dir, "experiments")
        experiment_configs = [file for file in os.listdir(experiments_dir) if file.endswith(".json")]
        for experiment_config in experiment_configs:
            relative_experiment_config_path = os.path.join("experiments", experiment_config)
            self.experiment_configs[experiment_config] = load_json_config_as_dict(self.config_top_dir,
                                                                                  relative_experiment_config_path)

    def get_experiment_names(self):
        return list(self.experiment_configs.keys())

    def get_experiment_by_name(self, name, do_resume=None, do_log_grads=None, dataset_file_path=None):
        json_name = name + ".json"
        if json_name in self.experiment_configs:
            config = self.experiment_configs[json_name]
            config["resume"] = do_resume
            config["log_grads"] = do_log_grads
            config["dataset"]["params"]["dataset_file_path"] = dataset_file_path
            return config
        err_msg = "ExperimentConfigurations {0} was not found. " \
                  "Available experiment configurations:\n{1}".format(json_name,
                                                                     "\n".join(
                                                                         sorted([n.replace(".json", "") for n in
                                                                                 self.experiment_configs.keys()])))
        raise Exception(err_msg)
