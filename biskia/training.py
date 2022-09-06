"""
Created on 24.08.2020

@author: Philipp
"""
from biskia import __load_cometml_experiment, __load_device, __load_model_from_config, __load_dataset_from_config, \
    __collate_variable_sequences, __log_parameters, __load_loss_fn, Saver, __load_interpreter
from biskia.callbacks import AverageLossMetric, CategoricalTextAccuracyMetric, GradientMeter, CallbackRegistry, \
    InterpreterCallbackRegistry, AverageEuclideanDistanceInterpreterMetric, ParaphraseAccuracy, \
    SemanticsTextGenerationMeter, AverageMetricsMetric, CategoricalInterpreterAccuracyMetric, BLEUScore
import torch
from torch import optim
from torch import nn
from torch.utils import data
import logging

logger = logging.getLogger(__file__)

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def perform_training_series_or_single(experiment_config):
    """
        @param experiment_config: a dictionary with all meta-information to perform the training
    """
    if "series" in experiment_config:
        perform_training_series(experiment_config)
    else:
        perform_training_single(experiment_config)


def perform_training_series(experiment_series_config):
    """
        @param experiment_series_config: a list of experiment configs (from a series configuration)
    """
    logger.info("Perform training for the experiment series '%s' ", experiment_series_config["name"])
    for experiment_config in experiment_series_config["series"]:
        perform_training_single(experiment_config)


def perform_training_single(experiment_config):
    """
        @param experiment_config: a dictionary with all meta-information to perform the training
    """
    logger.info("Perform training for the experiment '%s' ", experiment_config["name"])
    epoch_start = 1

    """ Load and setup the cometml experiment """
    experiment = __load_cometml_experiment(experiment_config["cometml"], experiment_config["name"],
                                           experiment_config["tags"])
    __log_parameters(experiment, experiment_config)

    """ Load and setup the model """
    device = __load_device(experiment_config["params"]["cpu_only"])
    model = __load_model_from_config(experiment_config["model"], experiment_config["task"])
    if experiment_config["resume"]:
        checkpoint = Saver.load_checkpoint(model, experiment_config["checkpoint_dir"], experiment_config["name"])
        epoch_start = checkpoint["checkpoint_epoch"] + 1
    model.to(device)

    """ Load and setup the optimizer """
    optimizer = optim.Adam(model.parameters())
    if experiment_config["resume"]:
        optimizer.load_state_dict(checkpoint["optimizer"])

    """ Load and setup the loss function """
    if "loss_fn" in experiment_config["params"]:
        loss_fn = __load_loss_fn(experiment_config["params"]["loss_fn"])
    else:
        # Mask padding_value=0 for loss computation
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    """ Load the training and validation data provider """
    dataset_train = __load_dataset_from_config(experiment_config["dataset"], experiment_config["task"], "train", device)
    dataloader_train = data.dataloader.DataLoader(dataset_train, batch_size=experiment_config["params"]["batch_size"],
                                                  shuffle=True, collate_fn=__collate_variable_sequences)

    dataset_validate = __load_dataset_from_config(experiment_config["dataset"], experiment_config["task"], "dev",
                                                  device)
    dataloader_validate = data.dataloader.DataLoader(dataset_validate,
                                                     batch_size=experiment_config["params"]["batch_size"],
                                                     shuffle=False, collate_fn=__collate_variable_sequences)

    """ Load interpreters and register interpreter callbacks """
    block_length = experiment_config["task"]["block_length"]

    interpreter = __load_interpreter(experiment_config["interpreter"], experiment_config["interpreter_name"], experiment_config, device)

    if experiment_config["task"]["input_type"] == "semantics":
        interpreter_callbacks = InterpreterCallbackRegistry({"semantics": interpreter}, experiment_config["task"], device, on_phase="validate")
        for idx, context in zip([0, 1, 2], ["source", "reference", "direction"]):
            interpreter_callbacks["avg_acc_" + context] = CategoricalInterpreterAccuracyMetric("semantics", experiment, index=idx, context=context)

        paraphrase_acc = ParaphraseAccuracy(experiment, {"semantics": interpreter}, experiment_config["task"], device, on_phase="validate")
        for idx, context in zip([0, 1, 2], ["source", "reference", "direction"]):
            paraphrase_acc["avg_acc_" + context] = CategoricalInterpreterAccuracyMetric("semantics", experiment, index=idx, context=context)
    else:
        interpreter_callbacks = InterpreterCallbackRegistry({"locations": interpreter}, experiment_config["task"], device, on_phase="validate")
        for idx, context in zip([0, 1], ["source_location", "target_location"]):
            interpreter_callbacks["avg_euclid_bl_" + context] = AverageEuclideanDistanceInterpreterMetric("locations", experiment, block_length, index=idx, context=context)

    """ Register callbacks and saver for training and validation """
    callbacks = CallbackRegistry()
    callbacks["interpreter_callbacks"] = interpreter_callbacks
    callbacks["paraphrase_accuracy"] = paraphrase_acc

    # The average metrics metric has to come after, so that the values are already computed
    if experiment_config["task"]["input_type"] == "semantics":
        callbacks["avg_loss"] = AverageLossMetric(experiment, on_phase="train")
        callbacks["categorical_text_accuracy"] = CategoricalTextAccuracyMetric(experiment, on_phase="train")
        callbacks["text_generation_meter"] = SemanticsTextGenerationMeter(experiment, dataset_train)
        callbacks["bleu_score"] = BLEUScore(experiment, dataset_train.vocab, on_phase="validate")
        # We need to define the average metric on callbacks, other than interpreter callbacks,
        # otherwise the metrics metric would get the interpreter inputs.
        # The average metrics metric has to come after, so that the values are already computed
        callbacks["avg_acc"] = AverageMetricsMetric(experiment, "epoch_avg", metrics=[
            interpreter_callbacks["avg_acc_source"],
            interpreter_callbacks["avg_acc_reference"],
            interpreter_callbacks["avg_acc_direction"]
        ], on_phase="validate")

        if experiment_config["model"]["name"] == "vanilla-transformer":
            saver = Saver(experiment_config["checkpoint_dir"], experiment_config["name"],
                          experiment_config["model"]["name"], "avg_loss", mode="lowest")
        else:
            saver = Saver(experiment_config["checkpoint_dir"], experiment_config["name"],
                          experiment_config["model"]["name"], "avg_acc", mode="highest")
    else:
        callbacks["avg_loss"] = AverageLossMetric(experiment, on_phase="train")
        callbacks["avg_euclidean"] = AverageMetricsMetric(experiment, "epoch_avg_euclid_bl", metrics=[
            interpreter_callbacks["avg_euclid_bl_source_location"],
            interpreter_callbacks["avg_euclid_bl_target_location"]
        ], on_phase="validate")
        if experiment_config["log_grads"]:
            callbacks["grads"] = GradientMeter(experiment, model.named_parameters())
        saver = Saver(experiment_config["checkpoint_dir"], experiment_config["name"],
                      experiment_config["model"]["name"], "avg_loss", mode="lowest")

    if len(callbacks) == 0:
        logger.info("No callbacks or saver registered!")

    """ Perform the training and validation """
    total_epochs = experiment_config["params"]["num_epochs"]
    for epoch in range(epoch_start, total_epochs + 1):
        if experiment_config["model"]["name"] == "vanilla-transformer":
            __train_transformer(device, dataloader_train, model, loss_fn, optimizer, epoch, experiment, callbacks)
            __validate_transformer(device, dataloader_validate, model, loss_fn, epoch, experiment, callbacks)
        else: # the model is lstm
            __train_lstm(device, dataloader_train, model, loss_fn, optimizer, epoch, experiment, callbacks)
            __validate_lstm(device, dataloader_validate, model, loss_fn, epoch, experiment, callbacks)
        if saver:
            saver.save_checkpoint_if_best(model, optimizer, epoch, callbacks)
    logger.info("Finished training for the experiment '%s' ", experiment_config["name"])


def __validate_lstm(device, dataloader_validate, model, loss_fn, current_epoch, experiment, callbacks):
    model.eval()
    with experiment.validate(), torch.no_grad():
        callbacks.on_epoch_start(phase="validate", epoch=current_epoch)
        for current_step, (batch_inputs, batch_labels) in enumerate(dataloader_validate):
            print("Validate epoch %s: Step %s" % (current_epoch, current_step), end="\r")
            batch_outputs, batch_inputs = model.generate(batch_inputs, device, beam_width=1, return_all=False)

            # Calculatingt the loss is misleading here, because it is very unlikely, that the network
            # produces five different and matching instructions, when the input coords and states are the same.
            # In the dataset there are five instructions per state and coords.

            callbacks.on_step(inputs=batch_inputs, outputs=batch_outputs, labels=batch_labels, mask=None, loss=None,
                              step=current_step)
        print()
        callbacks.on_epoch_end(epoch=current_epoch)

def __validate_transformer(device, dataloader_validate, model, loss_fn, current_epoch, experiment, callbacks):
    model.eval()
    with experiment.validate(), torch.no_grad():
        callbacks.on_epoch_start(phase="validate", epoch=current_epoch)
        for current_step, (batch_inputs, batch_labels) in enumerate(dataloader_validate):
            print("Validate epoch %s: Step %s" % (current_epoch, current_step), end="\r")
            # import pdb
            # pdb.set_trace()
            batch_outputs = model.greedy_decode(batch_inputs, max_len=50, device=device)
            # batch_outputs =  [[i.tolist() for i in batch_outputs], batch_inputs] #align outputs with __validate_lstm
            # batch_outputs = [batch_inputs, batch_outputs]
            callbacks.on_step(inputs=batch_inputs, outputs=batch_outputs, labels=batch_labels, mask=None, loss=None,
                              step=current_step)
        print()
        callbacks.on_epoch_end(epoch=current_epoch)

def __train_lstm(device, dataloader_train, model, loss_fn, optimizer, current_epoch, experiment, callbacks):
    model.train()
    with experiment.train():
        callbacks.on_epoch_start(phase="train", epoch=current_epoch)
        for current_step, (inputs, batch_labels) in enumerate(dataloader_train):
            print("Train epoch %s: Step %s" % (current_epoch, current_step), end="\r")
            optimizer.zero_grad()
            batch_outputs = model(inputs, device)
            # Output is B x L x V, but need B x V x L for loss
            batch_outputs = batch_outputs.permute(dims=[0, 2, 1])
            # Variable length instructions as labels
            batch_labels = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True).to(device)
            # Create mask for labels
            mask = torch.where(batch_labels > 0, torch.tensor(1, device=device), torch.tensor(0, device=device))
            loss = loss_fn(batch_outputs, batch_labels)
            loss.backward()
            optimizer.step()

            callbacks.on_step(inputs=inputs, outputs=batch_outputs, labels=batch_labels, mask=mask,
                              loss=loss, step=current_step)
        print()
        callbacks.on_epoch_end(epoch=current_epoch)

def __train_transformer(device, dataloader_train, model, loss_fn, optimizer, current_epoch, experiment, callbacks):
    model.train()
    with experiment.train():
        callbacks.on_epoch_start(phase="train", epoch=current_epoch)
        for current_step, (inputs, batch_labels) in enumerate(dataloader_train):
            print("Train epoch %s: Step %s" % (current_epoch, current_step), end="\r")
            optimizer.zero_grad()
            src = [item["semantics"] for item in inputs]
            src = torch.stack(src)
            src_mask = None

            tgt = [item["instruction"] for item in inputs]
            tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
            tgt_seq_len = tgt.shape[1]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len)

            enc_dec_output = model(src, tgt, src_mask, tgt_mask)

            prediction = model.model.generator(enc_dec_output)
            prediction = prediction.permute(0,2,1)

            batch_labels = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True).to(device)
            mask = torch.where(batch_labels > 0, torch.tensor(1, device=device), torch.tensor(0, device=device))
            loss = loss_fn(prediction, batch_labels)
            loss.backward()
            optimizer.step()

            callbacks.on_step(inputs=inputs, outputs=prediction, labels=batch_labels, mask=mask,
                              loss=loss, step=current_step)
        print()
        callbacks.on_epoch_end(epoch=current_epoch)
