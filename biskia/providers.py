"""
Created on 20.08.2020

@author: Philipp
"""
import collections

import torch
import os
import json


def determine_sub_directory(directory, sub_directory_name, to_read=True):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
        @param to_read: check for file existence, when true
    """
    sub_directory = os.path.join(directory, sub_directory_name)
    if os.path.isdir(directory):
        if sub_directory_name is None:
            raise Exception("Cannot determine sub-directory without sub_directory_name")
    if to_read and not os.path.isdir(sub_directory):
        raise Exception("There is no such sub-directory to read: " + sub_directory)
    return sub_directory


def exists_sub_directory(directory, sub_directory_name):
    sub_directory = os.path.join(directory, sub_directory_name)
    return exists_directory(sub_directory)


def exists_directory(directoy):
    if os.path.exists(directoy):
        return True
    return False


def determine_file_path(directory_or_file, lookup_filename=None, to_read=True):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
        @param to_read: check for file existence, when true
    """
    file_path = directory_or_file
    if os.path.isdir(directory_or_file):
        if lookup_filename is None:
            raise Exception("Cannot determine source file in directory without lookup_filename")
        file_path = os.path.join(directory_or_file, lookup_filename)
    if to_read and not os.path.isfile(file_path):
        raise Exception("There is no such file in the directory to read: " + file_path)
    return file_path


def load_json_from(directory_or_file, lookup_filename=None):
    """
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename)
    # print("Loading JSON from " + file_path)
    with open(file_path) as json_file:
        json_content = json.load(json_file)
    return json_content


def store_json_to(json_content, directory_or_file, lookup_filename=None):
    """
        @param json_content: the json data to store
        @param directory_or_file: to look for the source file
        @param lookup_filename: the filename to look for when a directory is given
    """
    file_path = determine_file_path(directory_or_file, lookup_filename, to_read=False)
    print("Persisting JSON to " + file_path)
    with open(file_path, "w") as json_file:
        json.dump(json_content, json_file, indent=4, sort_keys=True)
    return file_path


class BlocksLocationsDataset(object):

    def __init__(self, task, split_name, params, device):
        self.device = device
        self.task = task
        self.vocab = load_json_from(params["dataset_directory"], params["vocab"])
        """ Prepare (virtual) start and end token """
        self.start_token = task["start_token"]
        self.end_token = task["end_token"]
        self.vocab["word_to_id"]["<s>"] = self.start_token
        self.vocab["word_to_id"]["<e>"] = self.end_token
        self.vocab["id_to_word"][str(self.start_token)] = "<s>"
        self.vocab["id_to_word"][str(self.end_token)] = "<e>"
        if split_name == "file":
            self.samples = load_json_from(params["dataset_file_path"])
            self.__prepare_samples()
        else:
            self.samples = load_json_from(params["dataset_directory"], params["splits"][split_name])

    def __prepare_samples(self):
        """ Convert human input to model input"""
        pass

    def __getitem__(self, index):
        """
            @param index: a number in [0, length]
        """
        if "instruction_tokens_ids" in self.samples[index]:
            """ Prepare instructions and world states """
            encoded_words_raw = self.samples[index]["instruction_tokens_ids"]
            # int64 is preferred to potentially perform one_hot encoding later in pytorch
            # We want to predict the next word, so we append <eos> to the end
            labels = torch.as_tensor(encoded_words_raw + [self.end_token], dtype=torch.int64, device=self.device)
            # And pre-pend <s> to the provided instructions (This should actually be done in the data preparation.)
            instructions = torch.as_tensor([self.start_token] + encoded_words_raw, dtype=torch.int64,
                                           device=self.device)
        else:
            instructions = []
            labels = []
        """ Prepare world states """
        world_state_raw = self.samples[index]["world_state"]
        world_state = torch.as_tensor(world_state_raw, dtype=torch.float32, device=self.device)
        """ Prepare semantics """
        source_block = self.samples[index]["source_human"]
        reference_block = self.samples[index]["reference_human"]
        direction = self.samples[index]["direction_human"]
        decoration = torch.as_tensor(int(self.samples[index]["decoration"]), dtype=torch.long, device=self.device)
        """ Prepare locations"""
        source_location = torch.as_tensor(self.samples[index]["source_location"],
                                          dtype=torch.float32, device=self.device)
        target_location = torch.as_tensor(self.samples[index]["target_location"],
                                          dtype=torch.float32, device=self.device)
        return {"locations": torch.stack([source_location, target_location]),
                "source_block": source_block,
                "reference_block": reference_block,
                "direction": direction,
                "decoration": decoration,
                "world_state": world_state,
                "instruction": instructions}, labels

    def __len__(self):
        return len(self.samples)

    def get_label_names(self):
        return "instructions"

    def is_start_token(self, word_id):
        return word_id == self.start_token

    def is_end_token(self, word_id):
        return word_id == self.end_token

    def convert_to_word(self, word_id):
        if word_id == 0:
            return "<pad>"
        return self.vocab["id_to_word"][str(word_id)]

    def convert_to_name(self, coords):
        # return " ".join(["%s: %.2f" % (n, c) for (n, c) in zip(self.label_names, coords)])
        return " ".join(["%.2f" % c for c in coords])


class BlocksSemanticsDataset(object):

    def __init__(self, task, split_name, params, device):
        self.device = device
        self.task = task
        self.vocab = load_json_from(params["dataset_directory"], params["vocab"])
        """ Prepare (virtual) start and end token """
        self.start_token = task["start_token"]
        self.end_token = task["end_token"]
        self.vocab["word_to_id"]["<s>"] = self.start_token
        self.vocab["word_to_id"]["<e>"] = self.end_token
        self.vocab["id_to_word"][str(self.start_token)] = "<s>"
        self.vocab["id_to_word"][str(self.end_token)] = "<e>"
        if split_name == "file":
            self.samples = load_json_from(params["dataset_file_path"])
            self.__prepare_samples()
        else:
            self.samples = load_json_from(params["dataset_directory"], params["splits"][split_name])

    def __prepare_sample(self, sample):
        if str(sample["reference"]).isdigit():  # we handle both int-like and str-like digit inputs
            sample["reference"] = int(sample["reference"]) - 1
        else:
            sample["reference"] = self.task["brands_to_encodings"][sample["reference"]] - 1
        if str(sample["source"]).isdigit():  # we handle both int-like and str-like digit inputs
            sample["source"] = int(sample["source"]) - 1
        else:
            sample["source"] = self.task["brands_to_encodings"][sample["source"]] - 1
        sample["decoration"] = self.task["decorations_to_encodings"][sample["decoration"]]
        sample["direction"] = self.task["directions_to_encodings"][sample["direction"]]

    def __prepare_samples(self):
        [self.__prepare_sample(sample) for sample in self.samples]

    def __getitem__(self, index):
        """
            @param index: a number in [0, length]
        """
        if "instruction_tokens_ids" in self.samples[index]:
            """ Prepare instructions and world states """
            encoded_words_raw = self.samples[index]["instruction_tokens_ids"]
            # int64 is preferred to potentially perform one_hot encoding later in pytorch
            # We want to predict the next word, so we append <eos> to the end
            labels = torch.as_tensor(encoded_words_raw + [self.end_token], dtype=torch.int64, device=self.device)
            # And pre-pend <s> to the provided instructions (This should actually be done in the data preparation.)
            instructions = torch.as_tensor([self.start_token] + encoded_words_raw, dtype=torch.int64,
                                           device=self.device)
        else:
            instructions = []
            labels = []
        """ Prepare semantics """
        source_block = torch.as_tensor(int(self.samples[index]["source"]), dtype=torch.long, device=self.device)
        reference_block = torch.as_tensor(int(self.samples[index]["reference"]), dtype=torch.long, device=self.device)
        direction = torch.as_tensor(int(self.samples[index]["direction"]), dtype=torch.long, device=self.device)
        decoration = torch.as_tensor(int(self.samples[index]["decoration"]), dtype=torch.long, device=self.device)
        return {"semantics": torch.stack([source_block, reference_block, direction, decoration]),
                "world_state": [],  # For now, this is empty here
                "instruction": instructions}, labels

    def __len__(self):
        return len(self.samples)

    def get_label_names(self):
        return "instructions"

    def is_start_token(self, word_id):
        return word_id == self.start_token

    def is_end_token(self, word_id):
        return word_id == self.end_token

    def convert_to_word(self, word_id):
        if word_id == 0:
            return "<pad>"
        return self.vocab["id_to_word"][str(word_id)]

    def convert_to_name(self, encoding, label_name):
        if label_name == "direction":
            return self.task["encodings_to_directions"][str(encoding)]
        if label_name == "decoration":
            return self.task["encodings_to_decorations"][str(encoding)]
        return str(encoding)
