"""
Created on 24.08.2020

@author: Philipp
"""
from biskia import __load_cometml_experiment, __load_device, __load_model_from_config, __load_dataset_from_config, \
    __collate_variable_sequences, __log_parameters, __load_loss_fn, Saver, __load_interpreter
import torch
from torch.utils import data
import logging
from biskia.callbacks import CallbackRegistry, \
    SemanticsTextGenerationMeter, PlottingInterpreterMetric, InterpreterCallbackRegistry, \
    AverageEuclideanDistanceInterpreterMetric, AverageMetricsMetric, CategoricalInterpreterAccuracyMetric, BLEUScore

logger = logging.getLogger(__file__)


def perform_prediction_series_or_single(experiment_config, split_name):
    """
        @param experiment_config: a dictionary with all meta-information to perform the training
        @param split_name: Name of the dataset split on which to perform the prediction.
    """
    if "series" in experiment_config:
        perform_prediction_series(experiment_config, split_name)
    else:
        perform_prediction_single(experiment_config, split_name)


def perform_prediction_series(experiment_series_config, split_name):
    """
        @param experiment_series_config: a list of experiment configs (from a series configuration)
        @param split_name: Name of the dataset split on which to perform the prediction.
    """
    logger.info("Perform prediction for the experiment series '%s' on '%s'", experiment_series_config["name"],
                split_name)
    for experiment_config in experiment_series_config["series"]:
        perform_prediction_single(experiment_config, split_name)


def perform_prediction_single(experiment_config, split_name):
    """
        @param experiment_config: a dictionary with all meta-information to perform the training
        @param split_name: Name of the dataset split on which to perform the prediction.
    """
    logger.info("Perform prediction for the experiment '%s' on '%s'", experiment_config["name"], split_name)

    """ Load and setup the cometml experiment """
    experiment = __load_cometml_experiment(experiment_config["cometml"], experiment_config["name"],
                                           experiment_config["tags"])
    __log_parameters(experiment, experiment_config)

    """ Load and setup the model """
    device = __load_device(experiment_config["params"]["cpu_only"])
    model = __load_model_from_config(experiment_config["model"], experiment_config["task"])
    if "checkpoint_dir" in experiment_config:
        Saver.load_checkpoint(model, experiment, experiment_config["checkpoint_dir"], experiment_config["name"])
    model.to(device)

    """ Load the data provider """
    dataset_predict = __load_dataset_from_config(experiment_config["dataset"], experiment_config["task"], split_name,
                                                 device)
    dataloader_predict = data.dataloader.DataLoader(dataset_predict,
                                                    batch_size=experiment_config["params"]["batch_size"],
                                                    shuffle=False, collate_fn=__collate_variable_sequences)

    """ Load interpreters and register interpreter callbacks """
    block_length = experiment_config["task"]["block_length"]
    interpreter = __load_interpreter(experiment_config["interpreter"], experiment_config["interpreter_name"],
                                     experiment_config, device)
    if experiment_config["task"]["input_type"] == "semantics":
        interpreter_callbacks = InterpreterCallbackRegistry({"semantics": interpreter}, experiment_config["task"],
                                                            device, on_phase="test")
        for idx, context in zip([0, 1, 2], ["source", "reference", "direction"]):
            interpreter_callbacks["avg_acc_" + context] = \
                CategoricalInterpreterAccuracyMetric("semantics", experiment, index=idx, context=context)
    else:
        interpreter_callbacks = InterpreterCallbackRegistry({"locations": interpreter}, experiment_config["task"],
                                                            device, on_phase="test")
        for idx, context in zip([0, 1], ["source_location", "target_location"]):
            interpreter_callbacks["avg_euclid_bl_" + context] = \
                AverageEuclideanDistanceInterpreterMetric("locations", experiment, block_length,
                                                          index=idx, context=context)
            interpreter_callbacks["plot_" + context] = \
                PlottingInterpreterMetric("locations", experiment, dataset_predict, block_length,
                                          log_individual_plots=True, index=idx, context=context)

    """ Register callbacks """
    callbacks = CallbackRegistry()
    callbacks["interpreter_callbacks"] = interpreter_callbacks
    # The average metrics metric has to come after, so that the values are already computed
    if experiment_config["task"]["input_type"] == "semantics":
        callbacks["texts"] = SemanticsTextGenerationMeter(experiment, dataset_predict, store_max=100)
        callbacks["bleu"] = BLEUScore(experiment, dataset_predict.vocab)
        callbacks["avg_acc"] = AverageMetricsMetric(experiment, "epoch_avg_acc", metrics=[
            interpreter_callbacks["avg_acc_source"],
            interpreter_callbacks["avg_acc_reference"],
            interpreter_callbacks["avg_acc_direction"]
        ])
    else:
        callbacks["avg_euclidean"] = AverageMetricsMetric(experiment, "epoch_avg_euclid_bl", metrics=[
            interpreter_callbacks["avg_euclid_bl_source_location"],
            interpreter_callbacks["avg_euclid_bl_target_location"]
        ])

    """ Perform the test """
    __predict(device, dataloader_predict, model, None, experiment, callbacks)
    logger.info("Finished prediction for the experiment '%s' ", experiment_config["name"])


def __predict(device, dataloader_predict, model, criterion, experiment, callbacks):
    model.eval()
    with experiment.test(), torch.no_grad():
        callbacks.on_epoch_start(phase="test", epoch=None)
        for current_step, (batch_inputs, batch_labels) in enumerate(dataloader_predict):
            batch_outputs, batch_inputs = model.generate(batch_inputs, device, beam_width=1, return_all=False)

            # Calculatingt the loss or accuracy is misleading here, because it is very unlikely, that the network
            # produces five different and matching instructions, when the input coords and states are the same.
            # In the dataset there are five instructions per state and coords.

            callbacks.on_step(inputs=batch_inputs, outputs=batch_outputs, labels=batch_labels, mask=None, loss=None,
                              step=current_step)
        callbacks.on_epoch_end(epoch=None)
