from biskia import __load_cometml_experiment, __load_device, __load_model_from_config, __load_dataset_from_config, \
__collate_variable_sequences, __log_parameters, __load_loss_fn, Saver, __load_interpreter
import torch
from torch.utils import data
import logging
from biskia.callbacks import AverageLossMetric, CategoricalTextAccuracyMetric, GradientMeter, CallbackRegistry, InterpreterCallbackRegistry, AverageEuclideanDistanceInterpreterMetric,  SemanticsTextGenerationMeter, AverageMetricsMetric, CategoricalInterpreterAccuracyMetric, BLEUScore
import torch
from torch import optim
from torch import nn
from torch.utils import data
import logging
from biskia.configuration import ExperimentConfigurations
import torch.nn.functional as F
import json
import pickle
import numpy as np
import argparse


class MockConfig():
    def __init__(self):
        self.config_directory_path = "/Users/lukaborec/Projects/IM/060_bisk_interpreter_app/configs"
        self.checkpoint_dir = "/Users/lukaborec/Projects/IM/checkpoints"
        self.dataset_dir = "/Users/lukaborec/Projects/IM/060_bisk_interpreter_data/data/blockworld"
        self.file_path = "F:/Development/data/blockworld/MNIST/semantics/Custom.json"
        self.user = "lukaborec"

mock_config = MockConfig()
configs = ExperimentConfigurations(mock_config.config_directory_path, mock_config.checkpoint_dir, mock_config.dataset_dir, mock_config.user)

transformer_experiment_config = configs.get_experiment_by_name("vanilla-transformer-from-semantics", dataset_file_path=mock_config.file_path)
lstm_experiment_config = configs.get_experiment_by_name("lstm-from-semantics-init-step", dataset_file_path=mock_config.file_path)

device = __load_device(transformer_experiment_config["params"]["cpu_only"])

transformer_model = __load_model_from_config(transformer_experiment_config["model"], transformer_experiment_config["task"])
lstm_model = __load_model_from_config(lstm_experiment_config["model"], lstm_experiment_config["task"])

experiment=None

Saver.load_checkpoint(transformer_model, experiment, transformer_experiment_config["checkpoint_dir"], transformer_experiment_config["name"])
Saver.load_checkpoint(lstm_model, experiment, lstm_experiment_config["checkpoint_dir"], lstm_experiment_config["name"])

transformer_model.to(device)
lstm_model.to(device)

lstm_model.eval()
transformer_model.eval()

# VOCAB
import json
vocab = json.loads(open('/Users/lukaborec/Projects/IM/060_bisk_interpreter_data/data/blockworld/MNIST/semantics/vocab.json', 'r').read())
vocab['id_to_word']['659'] = "<sos>"
vocab['id_to_word']['660'] = "<eos>"
start_token = '659'
end_token = '660'
def encode_output(words):
    return torch.LongTensor([vocab["word_to_id"][word] for word in words.split()])


def decode_output(ids):
    w = [vocab["id_to_word"][str(int(id_))] for id_ in ids]
    return ' '.join(w)

# load the interpreter model
transformer_experiment_config["interpreter"], lstm_experiment_config["interpreter"]
# this only loads the trained interpreter model

interpreter = __load_interpreter(transformer_experiment_config["interpreter"], transformer_experiment_config["interpreter_name"],
                                     transformer_experiment_config, device)

def prepare(prediction):
    """
        Align the vocabulary: Remove start tokens and break on end tokens.
        Otherwise the interpreters word embeddings will fail.
    """
    result = []
    for w in prediction:
        if w.equal(torch.tensor(659)):
            continue
        if w.equal(torch.tensor(660)):
            if len(result) == 0:
                """ The generator seems to start with the end token here, so we just ignore and go on """
                continue
            return torch.stack(result)
        else:
            result.append(w)
    return torch.stack(result)

parser = argparse.ArgumentParser(description='Parsing CLI arguments.')
parser.add_argument('--num_paraphrases', type=float, help='num paraphrases to generate', default=5)

args = parser.parse_args()

num_paraphrases = args.num_paraphrases

DEVICE = "cpu"

print(f"Running beam search experiments with num_paraphrases={num_paraphrases}")

# Models
# models = [("lstm_model", lstm_model), ("transformer_model", transformer_model)]
models = [("transformer_model", transformer_model)]


for model_tuple in models:
    dataset_train = __load_dataset_from_config(lstm_experiment_config["dataset"], lstm_experiment_config["task"], "train", device)
    dataloader_train = data.dataloader.DataLoader(dataset_train, batch_size=1,
                                                  shuffle=True, collate_fn=__collate_variable_sequences)

    dataset_validate = __load_dataset_from_config(lstm_experiment_config["dataset"], lstm_experiment_config["task"], "dev",
                                                  device)
    dataloader_validate = data.dataloader.DataLoader(dataset_validate,
                                                     batch_size=1,
                                                     shuffle=False, collate_fn=__collate_variable_sequences)

    dataset_test = __load_dataset_from_config(lstm_experiment_config["dataset"], lstm_experiment_config["task"], "test", device)
    dataloader_test = data.dataloader.DataLoader(dataset_test, batch_size=1,
                                                  shuffle=True, collate_fn=__collate_variable_sequences)
    # Data loaders and split names
    dataloaders = [("train", dataset_train), ("val", dataloader_validate), ("test", dataloader_test)]

    model_name = model_tuple[0]
    print(f"Running experiment with {model_name}")
    model = model_tuple[1]

    for data_loader in dataloaders:
        print(f"Running data_loader_{data_loader[0]}...")
        dataloader = data_loader[1]

        # Initialize variables
        total_correct = 0
        total_paraphrases = 0
        json_output = []

        for current_step, (inputs, outputs) in enumerate(dataloader):
            try:
                semantics = inputs['semantics'][:3]
            except:
                semantics = inputs[0]['semantics'][:3]
                inputs = inputs[0]
            label = outputs

            # Feed instruction into model to generate paraphrases
            # try:
            if model_name == "lstm_model":
                paraphrases = model.generate_beam([inputs], DEVICE, num_paraphrases, return_all=True)[0]
            elif model_name == "transformer_model":
                paraphrases = model.generate_beam([inputs], DEVICE, num_paraphrases, return_all=True)
                paraphrases = [sublist[1].squeeze(0) for sublist in paraphrases] # Reverse order in tuple
            # except:
            #     print("Problem processing the following semantics:", semantics)
            #     continue
            # Clean paraphrases for input into the interpreter model
            prepared_paraphrases = []
            for p in paraphrases:
                try:
                    prepared_paraphrases.append({"instruction" : prepare(p)})
                except KeyError:
                    prepared_paraphrases.append({"instruction" : prepare(p['instruction'])})

            # Extract semantics from the generated paraphrases
            for pp in prepared_paraphrases:
                semantics_predictions = interpreter([pp], 'cpu')
                predictions = [torch.argmax(p, -1) for p in semantics_predictions]
                equal = torch.equal(torch.tensor(predictions), semantics)
                if equal:
                    total_correct +=1


            json_output.append({"input":semantics, "outputs":prepared_paraphrases, "label":label})
            total_paraphrases += 5


        # Write to file
        folder = "/Users/lukaborec/Projects/IM/060_bisk_interpreter_app/decoding_experiments"
        file_name = f"{model_name}_beam_search_{data_loader[0]}"
        print(f"Writing to file: {folder}/{file_name}.pkl")
        with open(f"{folder}/{file_name}.pkl", "wb") as doc:
            pickle.dump(json_output, doc)

        print("Number of correct paraphrases (w.r.t semantics):", total_correct / total_paraphrases)
        print()
