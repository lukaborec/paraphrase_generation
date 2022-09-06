from collections import OrderedDict
from datasets import load_metric
import torch
import logging

from biskia.stateviewer import StateViewer

logger = logging.getLogger(__file__)


class Callback(object):
    """
    Base class for callbacks.
    """

    def on_epoch_start(self, phase, epoch):
        pass

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        pass

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        pass


class Metric(Callback):
    """
    Base class for callbacks that collect metrics
    """

    @torch.no_grad()
    def to_value(self):
        pass


class InterpreterMetric(Metric):
    """
    Base class for callbacks that collect interpreter metrics
    """

    @torch.no_grad()
    def get_label_name(self):
        """
        :return: either "target_location" or "source_location"
        """
        pass


class CallbackRegistry(Callback):
    """
        Register one or more callbacks for callback method invocation. Keeps the order of added callbacks.
    """

    def __init__(self):
        self.callbacks = OrderedDict()

    def on_epoch_start(self, phase, epoch):
        for c in self.callbacks.values():
            c.on_epoch_start(phase, epoch)

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        for c in self.callbacks.values():
            c.on_step(inputs, outputs, labels, mask, loss, step)

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        for c in self.callbacks.values():
            c.on_epoch_end(epoch)

    def __getitem__(self, key):
        return self.callbacks[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Callback):
            raise Exception("Value to add is no Callback, but %s" % value.__class__)
        self.callbacks[key] = value

    def __len__(self):
        return len(self.callbacks)


class InterpreterCallbackRegistry(CallbackRegistry):
    """
        A wrapper for interpreter metrics. Take the generated instructions and
        evaluates the interpreter to produce target and source coordinates.

        The resulting coordinates are provided to the wrapped metrics which either
        handle "source_locations" or "target_locations" by implementing get_label_name().

        Per default these callbacks are only applied on_phase="validate" b.c. of performance
    """

    def __init__(self, interpreters: dict, task, device, on_phase="validate"):
        super().__init__()
        self.interpreters = interpreters
        self.device = device
        self.start_token = torch.as_tensor(task["start_token"], dtype=torch.int64, device=device)
        self.end_token = torch.as_tensor(task["end_token"], dtype=torch.int64, device=device)
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase != self.current_phase:
            return
        super(InterpreterCallbackRegistry, self).on_epoch_start(phase, epoch)

    def __prepare(self, prediction):
        """
            Align the vocabulary: Remove start tokens and break on end tokens.
            Otherwise the interpreters word embeddings will fail.
        """
        result = []
        for w in prediction:
            if w.equal(self.start_token):
                continue
            if w.equal(self.end_token):
                if len(result) == 0:
                    """ The generator seems to start with the end token here, so we just ignore and go on """
                    continue
                return torch.stack(result)
            else:
                result.append(w)
        return torch.stack(result)

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase != self.current_phase:
            return
        """
        :param inputs: the instructions, coordinates and world states
        :param outputs: the predicted sentence
        """
        """ Prepare the generated instruction and world state for interpretation """
        if isinstance(outputs, list):  # list of variance length sequences (the whole batch)
            predictions = outputs
        else:
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.detach()
        world_states = [i["world_state"] for i in inputs]
        interpreter_inputs = []
        for p, w, i in zip(predictions, world_states, inputs):
            interpreter_input = {"instruction": self.__prepare(p), "world_state": w}
            if "source_block" in i:
                interpreter_input["source_block"] = i["source_block"]
            if "reference_block" in i:
                interpreter_input["reference_block"] = i["reference_block"]
            if "direction" in i:
                interpreter_input["direction"] = i["direction"]
            interpreter_inputs.append(interpreter_input)

        results = dict()
        for label_name in self.interpreters:
            """ Interpret generated instruction """
            interpreter_output = self.interpreters[label_name](interpreter_inputs, self.device)
            interpreter_output_true = torch.stack([i[label_name] for i in inputs])
            results[label_name] = (interpreter_output, interpreter_output_true.permute(1, 0))

        for c in self.callbacks.values():
            label_name = c.get_label_name()
            if label_name in results:
                c.on_step(interpreter_inputs, results[label_name][0], results[label_name][1], mask, loss, step)
                continue
            raise Exception("Cannot handle label name: " + c.get_label_name())

    def on_epoch_end(self, epoch):
        if self.on_phase != self.current_phase:
            return
        super(InterpreterCallbackRegistry, self).on_epoch_end(epoch)

class ParaphraseAccuracy(CallbackRegistry):
    def __init__(self, experiment, interpreters: dict, task, device, on_phase="validate"):
        super().__init__()
        self.experiment = experiment
        self.interpreters = interpreters
        self.device = device
        self.start_token = torch.as_tensor(task["start_token"], dtype=torch.int64, device=device)
        self.end_token = torch.as_tensor(task["end_token"], dtype=torch.int64, device=device)
        self.on_phase = on_phase
        self.current_phase = None
        self.success = 0
        self.total = 0

    def __prepare(self, prediction):
        """
            Align the vocabulary: Remove start tokens and break on end tokens.
            Otherwise the interpreters word embeddings will fail.
        """
        result = []
        for w in prediction:
            if w.equal(self.start_token):
                continue
            if w.equal(self.end_token):
                if len(result) == 0:
                    """ The generator seems to start with the end token here, so we just ignore and go on """
                    continue
                return torch.stack(result)
            else:
                result.append(w)
        return torch.stack(result)

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase != self.current_phase:
            return

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase != self.current_phase:
            return
        self.success = 0
        self.total = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase != self.current_phase:
            return
        """
        :param inputs: the instructions, coordinates and world states
        :param outputs: the predicted sentence
        """
        """ Prepare the generated instruction and world state for interpretation """
        import pdb
        pdb.set_trace
        predictions = outputs
        world_states = [i["world_state"] for i in inputs]
        interpreter_inputs = []
        for p, w, i in zip(predictions, world_states, inputs):
            interpreter_input = {"instruction": self.__prepare(p), "world_state": w}
            interpreter_inputs.append(interpreter_input)

        results = dict()
        for label_name in self.interpreters:
            """ Interpret generated instruction """
            interpreter_output = self.interpreters[label_name](interpreter_inputs, self.device)
            interpreter_output_true = torch.stack([i[label_name] for i in inputs])
            results[label_name] = (interpreter_output, interpreter_output_true.permute(1, 0))

        for c in self.callbacks.values():
            label_name = c.get_label_name()
            if label_name in results:
                one = torch.argmax(interpreter_output[0], -1)
                two = torch.argmax(interpreter_output[1], -1)
                three = torch.argmax(interpreter_output[2], -1)
                all_predictions = torch.stack([one, two, three])
                all_predictions = all_predictions.permute(1,0)
                true_labels = results[label_name][1][:3, :]
                true_labels = true_labels.permute(1,0)

                total = torch.all(torch.eq(all_predictions, true_labels),  dim=1)

                self.success += sum(total).item()
                self.total += len(total)

    def on_epoch_end(self, epoch):
        if self.on_phase != self.current_phase:
            return
        self.experiment.log_metric("epoch_paraphrase_total_accuracy", (self.success / self.total), step=epoch)


class AverageMetricsMetric(Metric):
    """
    Register multiple metrics and average their values on_epoch_end
    """

    def __init__(self, experiment, name, metrics: list, on_phase=None):
        self.experiment = experiment
        self.name = name
        self.metrics = metrics
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase

    @torch.no_grad()
    def to_value(self):
        total = len(self.metrics)
        return torch.true_divide(sum([metric.to_value() for metric in self.metrics]), total).item()

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.experiment.log_metric(self.name, self.to_value(), step=epoch)


class CategoricalTextAccuracyMetric(Metric):

    def __init__(self, experiment, on_phase=None):
        self.total = 0
        self.correct = 0
        self.experiment = experiment
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.total = 0
        self.correct = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        _, predicted = torch.max(outputs, 1)
        tokens = mask.sum().item()  # single value tensor item moves to cpu
        self.total += tokens
        masked_correct = (predicted == labels) * mask
        masked_correct = masked_correct.sum().item()  # single value tensor item moves to cpu
        self.correct += masked_correct

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.correct, self.total).item()

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.experiment.log_metric("epoch_correct", self.correct, step=epoch)
        self.experiment.log_metric("epoch_errors", (self.total - self.correct), step=epoch)
        self.experiment.log_metric("epoch_accuracy", self.to_value(), step=epoch)


class AverageLossMetric(Metric):

    def __init__(self, experiment, name="epoch_loss", on_phase=None, index=None, context=None):
        self.experiment = experiment
        self.index = index
        self.context = context
        self.name = name
        self.value = 0
        self.total = 0
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        self.value = 0
        self.total = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.index is None:
            self.value += loss.item()
        else:
            self.value += loss[self.index].item()
        self.total = step

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.value, self.total)

    def on_epoch_end(self, epoch):
        if self.on_phase:
            if self.on_phase != self.current_phase:
                return
        if self.context:
            with self.experiment.context_manager(self.current_phase + "_" + self.context):
                self.experiment.log_metric(self.name, self.to_value(), step=epoch)
        else:
            self.experiment.log_metric(self.name, self.to_value(), step=epoch)

class BLEUScore(Metric):
    def __init__(self, experiment, vocab, ngram_order=2, on_phase="validate"):
        self.experiment = experiment
        self.vocab = vocab
        self.bleu_metric = load_metric("bleu", max_order=ngram_order)
        self.on_phase = on_phase
        self.current_phase = None

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        self.scores = []
        # if self.on_phase != self.current_phase:
        #     self.scores = []
        #     return

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.on_phase != self.current_phase:
            return
        reference_instructions = []
        for input_ in inputs:
            decoded = [self.vocab['id_to_word'][str(int(tensor))] for tensor in input_["instruction"][1:]] # remove <s>
            reference_instructions.append([' '.join(decoded).split()])

        predicted_instructions = []
        for input_ in outputs:
            decoded = []
            for tensor in input_:
                # try:
                if str(int(tensor)) == '660':
                    break
                # except:
                #     import pdb
                #     pdb.set_trace()
                else:
                    decoded.append(self.vocab['id_to_word'][str(int(tensor))])
            predicted_instructions.append(' '.join(decoded).split())

        result = self.bleu_metric.compute(predictions=predicted_instructions, references=reference_instructions)

        self.scores.append(result["bleu"])

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.on_phase != self.current_phase:
            return
        avg_bleu = sum(self.scores) / len(self.scores)
        self.experiment.log_metric(name="BLEU-2", value=avg_bleu, step=epoch)

class SemanticsTextGenerationMeter(Callback):
    """
        Store the generated text snippets.
    """

    def __init__(self, experiment, vocab, store_max=10):
        """
        @param store_max: the maximal amount of item to keep. CometML seems to have problems with more then 100 items.
            CometML does then break the connection to prevent DoS attacks.
        @param vocab: To convert the encoded input to actual text.
        """
        self.experiment = experiment
        self.vocab = vocab
        self.store_max = store_max
        self.texts = []

    def on_epoch_start(self, phase, epoch):
        self.texts = []

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if len(self.texts) >= self.store_max:
            return
        if isinstance(outputs, list):
            # variable length outputs (already with word indicies)
            predictions = outputs
        else:
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.detach()
        pred_inputs = []
        for i in inputs:
            pred_input = dict()
            pred_input["semantics"] = i["semantics"]
            if "idx" in i:
                pred_input["idx"] = i["idx"]
            else:
                pred_input["idx"] = 0
            pred_inputs.append(pred_input)
        for pred_input, encoded_words in zip(pred_inputs, predictions):
            encoded_words = torch.LongTensor(encoded_words).cpu().numpy()
            if len(self.texts) >= self.store_max:
                break
            s = pred_input["semantics"]
            self.texts.append((pred_input["idx"], s[0].item(), s[1].item(), s[2].item(), s[3].item(), encoded_words))

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        for (idx, source, reference, direction, decoration, encoded_words) in self.texts:
            words = []
            for e in encoded_words:
                if self.vocab.is_end_token(e):
                    if len(words) == 0:
                        continue
                    break
                words.append(self.vocab.convert_to_word(e))
            words = " ".join(words)
            json_meta = {
                "idx": idx,
                "source": source + 1,
                "reference": reference + 1,
                "direction": self.vocab.convert_to_name(direction, label_name="direction"),
                "decoration": self.vocab.convert_to_name(decoration, label_name="decoration")
            }
            self.experiment.log_text(words, metadata=json_meta, step=epoch)


class GradientMeter(Callback):

    def __init__(self, experiment, model_parameters):
        """
        @param model_parameters: the model parameters with weights (only weights are logged)
        """
        self.model_parameters = list(model_parameters)
        self.experiment = experiment
        self.epoch_gradients = None
        self.phase = None
        self.total = 0

    @staticmethod
    def initial_gradients(model_parameters):
        gradients = dict()
        for parameter in model_parameters:
            if parameter[1].requires_grad:
                parameter_name = parameter[0]
                if 'weight' in parameter_name:
                    gradients["grad." + parameter_name] = torch.zeros_like(parameter[1])
        return gradients

    def on_epoch_start(self, phase, epoch):
        self.phase = phase
        if self.phase != "train":
            return
        self.epoch_gradients = GradientMeter.initial_gradients(self.model_parameters)
        self.total = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.phase != "train":
            return
        # if (step == 0) or (step % 100 != 0):
        #    return
        self.total = self.total + 1
        for parameter in self.model_parameters:
            if parameter[1].requires_grad:
                parameter_name = parameter[0]
                if 'weight' in parameter_name:
                    grad_value = parameter[1].grad
                    self.epoch_gradients["grad." + parameter_name].add_(grad_value.abs())

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.phase != "train":
            return
        for name, grads in self.epoch_gradients.items():
            if len(grads) == 0:
                logger.warning("No gradients for %s", name)
                return
            grad_avg = torch.true_divide(torch.sum(grads), self.total)
            self.experiment.log_metric(name="avg." + name, value=grad_avg.cpu().numpy(), step=epoch)

            # For hist we sum over the steps and divide by step count
            # This is mirroring the negative values onto the positive axis (by using grad.abs())
            # Thus the resulting plot only gives an estimate about the scale of the gradients
            grad_hist_avg = torch.true_divide(grads, self.total)
            self.experiment.log_histogram_3d(values=grad_hist_avg.cpu().numpy(), name="avg." + name, step=epoch)


"""
Interpreter Metrics
"""

class CategoricalInterpreterAccuracyMetric(InterpreterMetric):

    def __init__(self, label_name, experiment, index=None, context=None):
        self.label_name = label_name
        self.experiment = experiment
        self.index = index
        self.context = context
        self.total = 0
        self.correct = 0

    @torch.no_grad()
    def get_label_name(self):
        return self.label_name

    @torch.no_grad()
    def on_epoch_start(self, phase, epoch):
        self.total = 0
        self.correct = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if self.index is not None:
            outputs = outputs[self.index]
            labels = labels[self.index]
        _, predicted = torch.max(outputs, 1)
        self.total += labels.size(0)  # single value tensor item moves to cpu
        self.correct += (predicted == labels).sum().item()  # single value tensor item moves to cpu

    @torch.no_grad()
    def to_value(self):
        return torch.true_divide(self.correct, self.total).item()

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.context:
            with self.experiment.context_manager(self.context):
                self.experiment.log_metric("epoch_correct", self.correct, step=epoch)
                self.experiment.log_metric("epoch_errors", (self.total - self.correct), step=epoch)
                self.experiment.log_metric("epoch_accuracy", self.to_value(), step=epoch)
        else:
            self.experiment.log_metric("epoch_correct", self.correct, step=epoch)
            self.experiment.log_metric("epoch_errors", (self.total - self.correct), step=epoch)
            self.experiment.log_metric("epoch_accuracy", self.to_value(), step=epoch)


class AverageEuclideanDistanceInterpreterMetric(InterpreterMetric):

    def __init__(self, label_name, experiment, block_length, index=None, context=None):
        self.label_name = label_name
        self.experiment = experiment
        self.block_length = block_length
        self.index = index
        self.context = context
        self.current_phase = None
        self.total = 0
        self.epoch_error = 0

    def on_epoch_start(self, phase, epoch):
        self.current_phase = phase
        self.total = 0
        self.epoch_error = 0

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        """
        :param outputs: the predicted coordinates
        :param labels: the true coordinates
        """
        if self.index is not None:
            outputs = outputs[self.index]
            labels = labels[self.index]
        step_error = (outputs - labels)
        step_error = step_error.pow(2)
        step_error = step_error.sum(dim=1)
        step_error = step_error.sqrt()
        step_error = step_error.sum()
        self.total += labels.size(0)
        self.epoch_error += step_error

    @torch.no_grad()
    def to_value(self):
        avg_euclid = torch.true_divide(self.epoch_error, self.total)
        avg_euclid_bl = torch.true_divide(avg_euclid, self.block_length)
        return avg_euclid_bl.cpu().numpy()

    @torch.no_grad()
    def get_label_name(self):
        return self.label_name

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.context:
            with self.experiment.context_manager(self.current_phase + "_" + self.context):
                self.experiment.log_metric("epoch_avg_euclid_bl", self.to_value(), step=epoch)
        else:
            self.experiment.log_metric("epoch_avg_euclid_bl", self.to_value(), step=epoch)


class PlottingInterpreterMetric(InterpreterMetric):

    def __init__(self, label_name, experiment, vocab, block_length, store_max=10,
                 log_individual_plots=False, index=None, context=None):
        """
        @param store_max: the maximal amount of item to keep. CometML seems to have problems with more then 100 items.
            CometML does then break the connection to prevent DoS attacks.
        @param vocab: To convert the encoded input to actual text.
        @param store_distance: Store only text which output prediction is outside distance to the ground truth.
        """
        self.label_name = label_name
        self.experiment = experiment
        self.viewer = StateViewer(experiment)
        self.vocab = vocab
        self.store_max = store_max
        self.block_length = block_length
        self.log_individual_plots = log_individual_plots
        self.samples = []
        self.current_phase = None
        self.index = index
        self.context = context

    @torch.no_grad()
    def get_label_name(self):
        return self.label_name

    def on_epoch_start(self, phase, epoch):
        self.samples = []
        self.current_phase = phase

    @torch.no_grad()
    def on_step(self, inputs, outputs, labels, mask, loss, step):
        if len(self.samples) >= self.store_max:
            return
        if self.index is not None:
            outputs = outputs[self.index]
            labels = labels[self.index]
        for interpreter_input, coords, coords_true in zip(inputs,
                                                          outputs.cpu().numpy(),
                                                          labels.cpu().numpy()):
            if len(self.samples) >= self.store_max:
                continue
            self.samples.append(
                {
                    "instruction": interpreter_input["instruction"].cpu().numpy(),
                    "world_state": interpreter_input["world_state"].cpu().numpy(),
                    "source_block": interpreter_input["source_block"],
                    "reference_block": interpreter_input["reference_block"],
                    "direction": interpreter_input["direction"],
                    "locations": coords,
                    "locations_gt": coords_true}
            )

    @torch.no_grad()
    def on_epoch_end(self, epoch):
        if self.context:
            with self.experiment.context_manager(self.current_phase + "_" + self.context):
                self.__perform_plotting(epoch)
        else:
            self.__perform_plotting(epoch)

    def __perform_plotting(self, epoch):
        for idx, sample in enumerate(self.samples):
            if self.log_individual_plots:
                words = []
                for e in sample["instruction"]:
                    if e == 0:
                        break
                    words.append(self.vocab.convert_to_word(e))
                words = " ".join(words)
                gt = " (%s,%s,%s)" % (sample["source_block"], sample["reference_block"], sample["direction"])
                words = words + gt
                x_p = sample["locations"][0]
                y_p = sample["locations"][2]
                x_t = sample["locations_gt"][0]
                y_t = sample["locations_gt"][2]
                self.viewer.plot(x_p, y_p, x_t, y_t, states=sample["world_state"], title=words,
                                 fig_name="sample_" + str(idx), epoch=epoch)

        x_p = [sample["locations"][0] for sample in self.samples]
        y_p = [sample["locations"][2] for sample in self.samples]
        x_t = [sample["locations_gt"][0] for sample in self.samples]
        y_t = [sample["locations_gt"][2] for sample in self.samples]
        self.viewer.plot(x_p, y_p, x_t, y_t, title=self.label_name, fig_name="accumulated", epoch=epoch)
