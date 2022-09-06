'''
Created on 24.08.2020

@author: Philipp
'''
from torch import nn
import torch


class MultiLocationsPredictor(nn.Module):

    def __init__(self, task, params):
        super(__class__, self).__init__()
        self.locations_source = LocationsPredictor(task, params)
        self.locations_target = LocationsPredictor(task, params)

    def forward(self, inputs, device):
        l1 = self.locations_source(inputs, device)
        l2 = self.locations_target(inputs, device)
        return l1, l2


class LocationsPredictor(nn.Module):

    def __init__(self, task, params):
        super(__class__, self).__init__()
        self.num_blocks = params["num_blocks"]
        self.num_directions = params["num_directions"]
        self.num_coordinates = params["num_coordinates"]
        self.multi_semantics_predictor = MultiSemanticsPredictor(task, params)

        self.direction_selector = nn.Linear(out_features=self.num_directions, in_features=20 + 20 + 9)
        self.directions_to_coords = nn.Linear(out_features=self.num_coordinates, in_features=self.num_directions)
        self.block_selector = nn.Linear(out_features=self.num_blocks, in_features=20 + 20 + 9)

    def forward(self, inputs, device):
        semantics = self.multi_semantics_predictor(inputs, device)
        semantics = torch.cat(semantics, dim=1)

        refblock = self.block_selector(semantics).softmax(dim=1)  # 32 x 20
        refblock = refblock.unsqueeze(dim=1)  # 32 x 1 x 20

        # 32 x 1 x 20 @ 32 x 20 x 3 = 32 x 1 x 3
        world_state = torch.stack([t["world_state"] for t in inputs])
        ref_coords = torch.matmul(refblock, world_state)
        ref_coords = torch.squeeze(ref_coords)  # 32 x 3

        direction = self.direction_selector(semantics).softmax(dim=1)  # 32 x 9
        offset = self.directions_to_coords(direction)  # 32 x 3

        return ref_coords + offset


class MultiSemanticsPredictor(nn.Module):

    def __init__(self, task, params):
        super(__class__, self).__init__()
        self.semantics_source = SemanticsPredictor(params["num_blocks"], params["hidden_size"], params["vocab_size"])
        self.semantics_reference = SemanticsPredictor(params["num_blocks"], params["hidden_size"], params["vocab_size"])
        self.semantics_direction = SemanticsPredictor(params["num_directions"], params["hidden_size"], params["vocab_size"])

    def forward(self, inputs, device):
        s1 = self.semantics_source(inputs, device)
        s2 = self.semantics_reference(inputs, device)
        s3 = self.semantics_direction(inputs, device)
        return s1, s2, s3


class SemanticsPredictor(nn.Module):

    def __init__(self, num_classes, hidden_size, vocab_size, dropout=0., bidirectional=False, num_lstm_layers=1):
        super(__class__, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_directions = 2 if self.bidirectional else 1
        self.num_classes = num_classes
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=self.hidden_size,
                                           padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True,
                            num_layers=self.num_lstm_layers, bidirectional=self.bidirectional, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=self.hidden_size * self.num_lstm_directions, out_features=self.num_classes)

    def forward(self, inputs, device):
        inputs = [sample["instruction"] for sample in inputs]  # Unpack data-dict
        batch_size = len(inputs)
        batch_lenghts = [len(sample) for sample in inputs]
        padded_x = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        encoded_padded_x = self.word_embedding(padded_x)
        encoded_padded_packed_x = nn.utils.rnn.pack_padded_sequence(encoded_padded_x, lengths=batch_lenghts,
                                                                    batch_first=True, enforce_sorted=False)
        encoded_padded_packed_x = encoded_padded_packed_x.to(device)
        init_c = torch.randn(self.num_lstm_layers * self.num_lstm_directions, batch_size, self.hidden_size).to(device)
        init_h = torch.randn(self.num_lstm_layers * self.num_lstm_directions, batch_size, self.hidden_size).to(device)
        outputs, _ = self.lstm(encoded_padded_packed_x, (init_h, init_c))
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # Collect the last output for a padded-sample sequence (the actually padded outputs are all zeros)
        # There is one sample output for each time-step in the sample B x L x H -> B x H
        last_outputs = torch.stack(
            [sample_outputs[sample_length - 1] for (sample_outputs, sample_length) in zip(outputs, batch_lenghts)])
        # B x H -> B x C
        last_outputs = self.dropout(last_outputs)
        logits = self.fc(last_outputs)
        return logits
