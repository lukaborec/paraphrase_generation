"""
Created on 24.08.2020

@author: Philipp
"""
import collections
import math

from torch import nn
import torch.nn.functional as F
import torch


class BeamSearchNode(object):
    def __init__(self, node_hidden_state, node_context_state, end_token, current_word=None):
        self.hx = node_hidden_state
        self.cx = node_context_state
        self.current_word = current_word  # is ignored during training
        self.output_word_logits = []  # necessary for training
        self.output_words = None
        self.end_token = end_token
        self.prob = None
        self.length = 1

    def __create_next(self, word_prob, word, end_token):
        next_node = BeamSearchNode(self.hx, self.cx, end_token, word)
        if self.output_words is None:
            next_node.output_words = word
        else:
            # here we cat, because we work only on a single sample
            next_node.output_words = torch.cat((self.output_words, word))
        if self.prob is None:
            next_node.prob = word_prob
        else:
            next_node.prob = self.prob + word_prob  # maximize the log-prob (least-neg.)
        next_node.length = self.length + 1
        return next_node

    def create_candidates(self, word_logits, beam_width, end_token):
        """
         Determine k candidate nodes for this node. Later only the top-k sequences are kept over all
         candidates of all nodes, so that we need at max k candidates for each node here
         (actually you could also keep all possible words as candidates).
        """
        log_probs = torch.log_softmax(word_logits, dim=1)
        candidate_probs, candidate_words = torch.topk(log_probs, k=beam_width, dim=1)
        candidates = [self.__create_next(candidate_probs[:, idx], candidate_words[:, idx], end_token)
                      for idx in range(beam_width)]
        return candidates

    def score(self):
        return torch.true_divide(self.prob, self.length)

    def is_done(self):
        if self.current_word is None:
            return False  # Batch is only done on max sequence length
        return torch.equal(self.current_word[0], self.end_token)

    def __repr__(self):
        return str(self)

    def __str__(self):  #
        if self.prob is None:
            return "Node(%.4f) %s" % (0., [])
        if self.output_words is None:
            return "Node(%.4f) %s" % (self.prob.item(), [])
        return "Node(%.4f) %s" % (self.prob.item(), [w.item() for w in self.output_words])

class LstmGenerator(nn.Module):
    def __init__(self, task, params):
        super(__class__, self).__init__()
        self.task = task
        self.num_classes = task["vocab_size"] + 1  # plus one for padding (start- and end-token already incl.)
        self.embedding_size = params["embedding_size"]
        self.hidden_size = params["hidden_size"]
        self.use_mode = params["use_mode"]
        self.use_states = params["use_states"]
        self.lstm_input_size = self.embedding_size
        self.start_token = torch.as_tensor(task["start_token"], dtype=torch.int64, device="cpu")
        self.end_token = torch.as_tensor(task["end_token"], dtype=torch.int64, device="cpu")

        if self.use_states:
            self.states_embedding = nn.Linear(in_features=20 * 3, out_features=self.embedding_size)
            self.lstm_input_size = self.lstm_input_size + self.embedding_size  # double the size

        if self.task["name"] == "locations-to-instructions":
            """
            self.source_block_embedding = nn.Linear(in_features=20 * 3, out_features=self.embedding_size)
            self.reference_block_embedding = nn.Linear(in_features=20 * 3, out_features=self.embedding_size)
            self.direction_embedding = nn.Linear(in_features=20 * 3, out_features=self.embedding_size)

            self.source_block_selector = nn.Linear(in_features=self.embedding_size, out_features=20)
            self.reference_block_selector = nn.Linear(in_features=self.embedding_size, out_features=20)
            self.direction_selector = nn.Linear(in_features=self.embedding_size, out_features=9)

            self.semantics_embedding = nn.Linear(in_features=20 + 20 + 9 + 2, out_features=self.embedding_size)
            """
            self.source_block_selector = nn.Linear(in_features=20, out_features=20)
            self.reference_block_selector = nn.Linear(in_features=20, out_features=20)
            self.direction_selector = nn.Linear(in_features=self.embedding_size, out_features=9)
            self.locations_embedding = nn.Linear(in_features=3, out_features=self.embedding_size)
            self.semantics_embedding = nn.Linear(in_features=20 + 20 + 9 + 2, out_features=self.embedding_size)

        if self.task["name"] == "semantics-to-instructions":
            self.semantics_embedding = nn.Linear(in_features=20 + 20 + 9 + 2, out_features=self.embedding_size)

        self.words_to_embeddings = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.embedding_size,
                                                padding_idx=0)

        self.cell = nn.LSTMCell(input_size=self.lstm_input_size, hidden_size=self.hidden_size)
        self.dropout = nn.Dropout(p=params["dropout"])
        self.output_predictor = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_size)
        self.word_predictor = nn.Linear(in_features=self.embedding_size, out_features=self.num_classes)

    def get_block_length(self):
        return self.task["block_length"]

    def __initialize_network(self, inputs, device):

        batch_size = len(inputs)
        hx = torch.randn(batch_size, self.hidden_size).to(device)
        cx = torch.randn(batch_size, self.hidden_size).to(device)

        state_embeddings = None
        if self.task["name"] == "locations-to-instructions":
            """
            # B x 20 x 3
            world_states = torch.stack([sample["world_state"] for sample in inputs])

            source_locations = torch.stack([sample["locations"][0] for sample in inputs])
            source_locations = source_locations.unsqueeze(dim=1)  # promote tensor to enable substract
            source_block_locations = torch.sub(world_states, source_locations)
            source_block_locations = source_block_locations * source_block_locations
            # should learn to select the block that is (closer to) zero
            source_block_locations = torch.flatten(source_block_locations, start_dim=1)
            source_block_embeddings = self.source_block_embedding(source_block_locations).relu()
            source_blocks = self.source_block_selector(source_block_embeddings)
            source_blocks = source_blocks.softmax(dim=1)

            target_locations = torch.stack([sample["locations"][1] for sample in inputs])
            target_locations = target_locations.unsqueeze(dim=1)  # promote tensor to enable substract
            reference_block_locations = torch.sub(world_states, target_locations)
            reference_block_locations = reference_block_locations * reference_block_locations
            # should learn to select the block that is closer to zero
            reference_block_locations = torch.flatten(reference_block_locations, start_dim=1)
            reference_block_embeddings = self.reference_block_embedding(reference_block_locations).relu()
            reference_blocks = self.reference_block_selector(reference_block_embeddings).softmax(dim=1)

            # the directions depends on the reference block
            direction_locations = reference_blocks.unsqueeze(dim=-1) * world_states
            direction_locations = torch.sub(world_states, direction_locations)
            direction_locations = torch.flatten(direction_locations, start_dim=1)
            direction_embeddings = self.direction_embedding(direction_locations).relu()
            directions = self.direction_selector(direction_embeddings).softmax(dim=1)
            """

            # B x 20 x 3
            world_states = torch.stack([sample["world_state"] for sample in inputs])
            world_states_embedding = self.locations_embedding(world_states).relu()

            source_locations = torch.stack([sample["locations"][0] for sample in inputs])
            source_locations_embedding = self.locations_embedding(source_locations).relu()
            source_locations_embedding = source_locations_embedding.unsqueeze(dim=-1)
            source_block_embeddings = torch.squeeze(world_states_embedding @ source_locations_embedding)
            source_blocks = self.source_block_selector(source_block_embeddings).relu()

            target_locations = torch.stack([sample["locations"][1] for sample in inputs])
            target_locations_embedding = self.locations_embedding(target_locations).relu()
            target_locations_embedding = target_locations_embedding.unsqueeze(dim=-1)
            reference_block_embeddings = torch.squeeze(world_states_embedding @ target_locations_embedding)
            reference_blocks = self.reference_block_selector(reference_block_embeddings).relu()

            direction_embeddings = torch.squeeze(reference_blocks.unsqueeze(dim=1) @ world_states_embedding)
            directions = self.direction_selector(direction_embeddings).relu()

            decorations = torch.stack([torch.nn.functional.one_hot(sample["decoration"], num_classes=2)
                                       for sample in inputs]).float()

            semantics = torch.cat([source_blocks, reference_blocks, directions, decorations], dim=1)
            input_embeddings = self.semantics_embedding(semantics)
        elif self.task["name"] == "semantics-to-instructions":
            semantics = [torch.cat([torch.nn.functional.one_hot(sample["semantics"][0], num_classes=20),
                                    torch.nn.functional.one_hot(sample["semantics"][1], num_classes=20),
                                    torch.nn.functional.one_hot(sample["semantics"][2], num_classes=9),
                                    torch.nn.functional.one_hot(sample["semantics"][3], num_classes=2)])
                         for sample in inputs]
            semantics = torch.stack(semantics).float()
            input_embeddings = self.semantics_embedding(semantics)
        else:
            raise Exception("Cannot handle task: " + self.task["name"])

        if self.use_mode == "init-step":
            if self.use_states:
                hx, cx = self.cell(torch.cat([input_embeddings, state_embeddings], dim=1), (hx, cx))
            else:
                hx, cx = self.cell(input_embeddings, (hx, cx))
        elif self.use_mode == "init-state":
            cx = input_embeddings
            hx = input_embeddings
        else:
            raise Exception("Cannot handle use_mode: " + self.use_mode)

        return hx, cx, state_embeddings

    def __step(self, node, input_embeddings):
        if self.use_states:  # input_embeddings should be a list [previous_word_embedding, state_embeddings]
            node.hx, node.cx = self.cell(torch.cat(input_embeddings, dim=1), (node.hx, node.cx))
        else:
            node.hx, node.cx = self.cell(input_embeddings, (node.hx, node.cx))
        # B x H -> B x E
        output_embeddings = self.output_predictor(self.dropout(node.hx))
        # B x E -> B x C
        word_logits = self.word_predictor(self.dropout(output_embeddings))
        return word_logits

    def forward(self, inputs, device):
        if not self.training:
            raise Exception("Cannot forward() in evaluation mode. Please perform model.train() before.")
        """ Prepare teacher forcing for training """
        word_encodings = [sample["instruction"] for sample in inputs]  # Unpack data-dict
        word_paddings = nn.utils.rnn.pad_sequence(word_encodings, batch_first=True)
        word_embeddings = self.words_to_embeddings(word_paddings).to(device) # e.g. 32,65 -> 32,65,300
        hx, cx, state_embeddings = self.__initialize_network(inputs, device)
        # For training we will only have a single node here and use teacher forcing
        node = BeamSearchNode(hx, cx, self.end_token)
        max_steps = word_embeddings.size()[1]
        for t in range(max_steps):  # longest sequence in batch
            output_word_logit = self.__step(node, word_embeddings[:, t, :]) # <- this is  teacher forcing
            # Only collect the logits (outputs words are not feeded back anyway)
            node.output_word_logits.append(output_word_logit)
        # L x B x H -> B x L x H
        outputs = torch.stack(node.output_word_logits)
        outputs = outputs.permute(1, 0, 2)
        return outputs

    def generate(self, batch_inputs, device, beam_width=3, return_all=False):
        if self.training:
            raise Exception("Cannot generate in training mode. Please perform model.eval() before.")
        if beam_width == 1:
            return self.generate_greedy(batch_inputs, device)
        return self.generate_beam(batch_inputs, device, beam_width, return_all)

    def generate_greedy(self, batch_inputs, device):
        if self.training:
            raise Exception("Cannot generate in training mode. Please perform model.eval() before.")
        hx, cx, state_embeddings = self.__initialize_network(batch_inputs, device)
        start_token = torch.full(size=(len(batch_inputs),), fill_value=self.start_token.item(), dtype=torch.int64,
                                 device=device)
        node = BeamSearchNode(hx, cx, self.end_token, start_token)
        node.output_words = []
        for t in range(self.task["max_length"]):
            previous_word_embedding = self.words_to_embeddings(node.current_word)
            output_word_logit = self.__step(node, previous_word_embedding)
            output_words = torch.argmax(output_word_logit, dim=1)
            node.current_word = output_words
            node.output_words.append(output_words)
        batch_outputs = torch.stack(node.output_words)
        batch_outputs = batch_outputs.permute(1, 0)
        batch_outputs = [output for output in batch_outputs]  # listify the return values

        # prune everything after the end token
        cleaned_output = []
        for prediction in batch_outputs:
            temp = []
            for tensor in prediction:
                temp.append(tensor.item())
                if tensor.item() == 660:
                    break
            cleaned_output.append(temp)

        return cleaned_output, batch_inputs

# ------------------------------------------------
    def nucleus_sampling(self, input, device, top_p, num_paraphrases=1, filter_value = 0):
        if self.training:
            raise Exception("Cannot generate in training mode. Please perform model.eval() before.")
        assert top_p > 0, "top_p parameter must be in range (0,1]."

        hx, cx, state_embeddings = self.__initialize_network(input, device)
        start_token = torch.full(size=(len(input),), fill_value=self.start_token.item(), dtype=torch.int64,
                         device=device)
        paraphrases = []

        # Generate a stopping criterion to break the while loop
        num_total_generated_paraphrases = 10
        current_paraphrase_count = 0

        while len(paraphrases) < num_paraphrases and current_paraphrase_count < num_total_generated_paraphrases:
            node = BeamSearchNode(hx, cx, self.end_token, start_token)
            node.output_words = []

            for t in range(self.task["max_length"]):
                previous_word_embedding = self.words_to_embeddings(node.current_word)
                output_word_logit = self.__step(node, previous_word_embedding)

                sorted_logits, sorted_indices = torch.sort(output_word_logit, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                output_word_logit[:, indices_to_remove] = 0

                try:
                    output_word = torch.multinomial(prob, 1)[0] # because torch.multinomial returns a list and we only want the first element
                except:  # RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
                    output_word = sorted_indices.squeeze()[0].unsqueeze(0)
                # except:
                #     print("RuntimeError: probability tensor contains either `inf`, `nan` or element < 0")
                #     # batch_outputs = torch.stack(node.output_words)
                #     # batch_outputs = batch_outputs.permute(1, 0)
                #     batch_outputs = [output for output in batch_outputs]  # listify the return values
                #     paraphrases.
                node.current_word = output_word
                node.output_words.append(output_word)

                if node.current_word.item() == self.end_token:
                    current_paraphrase_count += 1 # Increment count
                    # Save the paraphrase only if it hasn't been generated previously
                    exists = False
                    current_paraphrase = torch.cat(node.output_words)
                    for p in paraphrases:
                        if torch.equal(current_paraphrase, p):
                            exists = True
                    if not exists:
                        paraphrases.append(torch.cat(node.output_words))
                    break

        # batch_outputs = torch.stack(node.output_words)
        # batch_outputs = batch_outputs.permute(1, 0)
        # batch_outputs = [output for output in batch_outputs]  # listify the return values
        return paraphrases
# ------------------------------------------------
    def calculate_sentence_probability(self, word_list, inputs, device):
        # rename to calculate_sentence_log_likelihood
        start_token = torch.full(size=(1,), fill_value=self.start_token.item(), dtype=torch.int64, device='cpu')
        hx, cx, state_embeddings = self.__initialize_network(inputs, device)
        node = BeamSearchNode(hx, cx, self.end_token, start_token)

        sentence_log_prob = []

        previous_word_embedding = self.words_to_embeddings(node.current_word)
        output_word_logit = self.__step(node, previous_word_embedding)
        probs = F.log_softmax(output_word_logit, dim=-1)

        for word in word_list:
            sentence_log_prob.append(probs[0, word.item()].item())

            node.current_word = word

            previous_word_embedding = self.words_to_embeddings(node.current_word)
            if len(previous_word_embedding.shape) == 1:
                previous_word_embedding = previous_word_embedding.unsqueeze(0)
            output_word_logit = self.__step(node, previous_word_embedding)
            probs = F.log_softmax(output_word_logit, dim=-1)

        return sum(sentence_log_prob) / len(sentence_log_prob) * (-1)

    def get_per_timestep_probs(self, word_list, inputs, top_p, device):
        """
        word_list: a list of tokens/tensors of an already generated sequence
        inputs: the semantics input used to generate word_list
        top_p: nucleus sampling cumulative probability parameter

        This method returns the cumulative distribution for each token t.
        The output is used to inspect the model's behavior, debugging, etc.
        """
        # rename to calculate_sentence_log_likelihood
        start_token = torch.full(size=(1,), fill_value=self.start_token.item(), dtype=torch.int64, device='cpu')
        hx, cx, state_embeddings = self.__initialize_network(inputs, device)
        node = BeamSearchNode(hx, cx, self.end_token, start_token)

        timestep_dist = []

        previous_word_embedding = self.words_to_embeddings(node.current_word)
        output_word_logit = self.__step(node, previous_word_embedding)

        # nucleus sampling steps
        sorted_logits, sorted_indices = torch.sort(output_word_logit, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        fixed_output_word_logit = output_word_logit
        fixed_output_word_logit[:, indices_to_remove] = 0
        timestep_dist.append({"current_word": node.current_word.item(),
                            "sorted_logits": sorted_logits,
                            "sorted_indices": sorted_indices,
                            "fixed_output_word_logit": fixed_output_word_logit})

        probs = F.log_softmax(output_word_logit, dim=-1)

        for word in word_list:
            node.current_word = word

            previous_word_embedding = self.words_to_embeddings(node.current_word)

            if len(previous_word_embedding.shape) == 1:
                previous_word_embedding = previous_word_embedding.unsqueeze(0)

            output_word_logit = self.__step(node, previous_word_embedding)

            # nucleus sampling steps
            sorted_logits, sorted_indices = torch.sort(output_word_logit, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            fixed_output_word_logit = output_word_logit
            fixed_output_word_logit[:, indices_to_remove] = 0
            timestep_dist.append({"current_word": node.current_word.item(),
                                "sorted_logits": sorted_logits,
                                "sorted_indices": sorted_indices,
                                "fixed_output_word_logit": fixed_output_word_logit})

            probs = F.log_softmax(output_word_logit, dim=-1)

        return timestep_dist



    def generate_beam(self, batch_inputs, device, beam_width=3, return_all=False):
        if self.training:
            raise Exception("Cannot generate in training mode. Please perform model.eval() before.")
        batch_outputs = []
        for input_sample in batch_inputs:
            hx, cx, state_embeddings = self.__initialize_network([input_sample], device)
            # Generate and keep all initial candidates (otherwise we have the same sequence for each node)
            start_node = BeamSearchNode(hx, cx, self.end_token)
            start_token = torch.full(size=(1,), fill_value=self.start_token.item(), dtype=torch.int64, device=device)
            start_logits = self.__step(start_node, self.words_to_embeddings(start_token))
            nodes = start_node.create_candidates(start_logits, beam_width, self.end_token)
            for t in range(self.task["max_length"]):  # longest sequence to produce
                # We store the next words as candidate nodes which are at most k*k nodes
                candidates = [n for n in nodes if n.is_done()]  # all finished nodes stay candidates
                for node in nodes:
                    # Ended sequences are not expanded, but might be kept as candidates as you can see above
                    if node.is_done():
                        continue
                    previous_word_embedding = self.words_to_embeddings(node.current_word)
                    output_word_logit = self.__step(node, previous_word_embedding)
                    candidates.extend(node.create_candidates(output_word_logit, beam_width, self.end_token))
                # Sort candidates by sequence score and keep only the top-k sequences
                candidates_sorted = sorted(candidates, key=lambda item: item.score(), reverse=True)
                nodes = candidates_sorted[:beam_width]
                # Break when all nodes are finished
                if all([n.is_done() for n in nodes]):
                    break
            if not return_all:  # Return only the sequence with the highest score
                nodes = sorted(nodes, key=lambda item: item.score(), reverse=True)[:1]
            outputs = [node.output_words for node in nodes]
            batch_outputs.append(outputs)
        # For each input we might have k outputs
        flatten_outputs = []
        flatten_inputs = []
        for sample_output, sample_input in zip(batch_outputs, batch_inputs):
            flatten_outputs.extend(sample_output)
            for _ in range(len(sample_output)):
                flatten_inputs.append(sample_input)  # duplicate the inputs
        batch_outputs = flatten_outputs
        batch_inputs = flatten_inputs
        return batch_outputs, batch_inputs

    def paraphrase_with_end_state(self, batch_inputs, device, iterations=3):
        print("Uses prvious iteration's end state as the start hidden state for the next iteration...")
        if self.training:
            raise Exception("Cannot generate in training mode. Please perform model.eval() before.")
        batch_outputs = []
        for input_sample in batch_inputs:
            sample_outputs = []
            sample_end_states = []
            for iteration in range(iterations):
                if len(sample_end_states) == 0:
                    hx, cx, state_embeddings = self.__initialize_network([input_sample], device)
                else:
                    print("using previous last state...")
                    hx, cx = sample_end_states[-1]
                start_token = torch.full(size=(1,), fill_value=self.start_token.item(), dtype=torch.int64,
                                         device=device)
                node = BeamSearchNode(hx, cx, self.end_token, start_token)
                node.output_words = []
                for t in range(self.task["max_length"]):  # longest sequence to produce
                    previous_word_embedding = self.words_to_embeddings(node.current_word)
                    output_word_logit = self.__step(node, previous_word_embedding)
                    if iteration > 0:
                        self.__mask_prev_iter_word(sample_outputs, iteration, output_word_logit, t)
                    output_word = torch.argmax(output_word_logit, dim=1)
                    node.current_word = output_word
                    node.output_words.append(output_word)
                    if node.is_done():
                        break
                sample_outputs.append(torch.stack(node.output_words).squeeze())
                sample_end_states.append((node.hx, node.cx))
            batch_outputs.append(sample_outputs)
        # For each input we might have k*i outputs
        flatten_outputs = []
        flatten_inputs = []
        for sample_output, sample_input in zip(batch_outputs, batch_inputs):
            for idx, s in enumerate(sample_output):
                flatten_outputs.append(s)
                idx_input = sample_input.copy()
                idx_input["idx"] = idx + 1
                flatten_inputs.append(idx_input)
        return flatten_outputs, flatten_inputs


    def paraphrase(self, batch_inputs, device, iterations=3):
        """
            Like generate_greedy() but with several sequences per sample.
        """
        print("Paraphrasing...")
        if self.training:
            raise Exception("Cannot generate in training mode. Please perform model.eval() before.")
        """
            For paraphrasing we use a naive "start" from scratch with prohibition of previously said words.
            - An alternative approach would be to start from the end-token (elaborate) and keep the last state,
            but the network was not trained in this way and will probably produce only "garbage".
            - Another approach would be to add the last state to the initial state and start from "scratch".
            But still we would more likely want to train using the five different instructions in random order.

            I actually would understand paraphrasing rather as "adding jitter" to a sentence embedding.
            The actual training task would be to get the correct amount of jitter, so that
                - the sequence is not just repeated (wich can be a known constraint) (reward signal)
                - the meaning of the sentence changes too much (which can be controlled by the interpreter)
        """
        batch_outputs = []
        for input_sample in batch_inputs:
            sample_outputs = []
            for iteration in range(iterations):
                hx, cx, state_embeddings = self.__initialize_network([input_sample], device)
                start_token = torch.full(size=(1,), fill_value=self.start_token.item(), dtype=torch.int64,
                                         device=device)
                node = BeamSearchNode(hx, cx, self.end_token, start_token)
                node.output_words = []
                for t in range(self.task["max_length"]):  # longest sequence to produce
                    previous_word_embedding = self.words_to_embeddings(node.current_word)
                    output_word_logit = self.__step(node, previous_word_embedding)
                    if iteration > 0:
                        self.__mask_prev_iter_word(sample_outputs, iteration, output_word_logit, t)
                    output_word = torch.argmax(output_word_logit, dim=1)
                    node.current_word = output_word
                    node.output_words.append(output_word)
                    if node.is_done():
                        break
                sample_outputs.append(torch.stack(node.output_words).squeeze())
            batch_outputs.append(sample_outputs)
        # For each input we might have k*i outputs
        flatten_outputs = []
        flatten_inputs = []
        for sample_output, sample_input in zip(batch_outputs, batch_inputs):
            for idx, s in enumerate(sample_output):
                flatten_outputs.append(s)
                idx_input = sample_input.copy()
                idx_input["idx"] = idx + 1
                flatten_inputs.append(idx_input)
        return flatten_outputs, flatten_inputs

    def __mask_prev_iter_word(self, iter_outputs, iteration, output_word_logit, t):
        if iteration < 1:
            return
        for prev in range(iteration):
            pre_iter_sequence = iter_outputs[iteration - prev - 1]
            pre_iter_sequence_length = len(pre_iter_sequence)
            if pre_iter_sequence_length > t:  # allow to produce longer sequences
                prev_iter_word = pre_iter_sequence[t]
                if prev_iter_word.item() in [23, 24, 26, 27, 31, 32, 33, 34, 35, 513,
                                             515, 517, 518, 519, 520, 521, 522, 523, 526, 535]:
                    continue  # allow to repeat block numbers
                if torch.equal(prev_iter_word, self.end_token):  # allow to repeat end-tokens
                    continue
                output_word_logit[..., prev_iter_word] = - math.inf

    def elaborate(self, batch_inputs, device, iterations=3):
        """
            Like generate_greedy() but with several sequences per sample.
        """
        if self.training:
            raise Exception("Cannot generate in training mode. Please perform model.eval() before.")
        batch_outputs = []
        for input_sample in batch_inputs:
            sample_outputs = []
            """ We keep the state over all iterations """
            hx, cx, state_embeddings = self.__initialize_network([input_sample], device)
            start_token = torch.full(size=(1,), fill_value=self.start_token.item(), dtype=torch.int64,
                                     device=device)
            node = BeamSearchNode(hx, cx, self.end_token)
            for iteration in range(iterations):
                node.current_word = start_token
                node.output_words = []
                for t in range(self.task["max_length"]):  # longest sequence to produce
                    previous_word_embedding = self.words_to_embeddings(node.current_word)
                    output_word_logit = self.__step(node, previous_word_embedding)
                    output_word = torch.argmax(output_word_logit, dim=1)
                    node.current_word = output_word
                    node.output_words.append(output_word)
                    if node.is_done():
                        break
                sample_outputs.append(torch.stack(node.output_words).squeeze())
            batch_outputs.append(sample_outputs)
        # For each input we might have k*i outputs
        flatten_outputs = []
        flatten_inputs = []
        for sample_output, sample_input in zip(batch_outputs, batch_inputs):
            for idx, s in enumerate(sample_output):
                flatten_outputs.append(s)
                idx_input = sample_input.copy()
                idx_input["idx"] = idx + 1
                flatten_inputs.append(idx_input)
        return flatten_outputs, flatten_inputs

class AttentionalLstmGenerator(nn.Module):

    def __init__(self, task, params):
        super(__class__, self).__init__()
        self.task = task
        self.num_classes = task["vocab_size"] + 1  # plus one for padding (start- and end-token already incl.)
        self.embedding_size = params["embedding_size"]
        self.hidden_size = self.embedding_size + self.embedding_size
        self.start_token = torch.as_tensor(task["start_token"], dtype=torch.int64, device="cpu")
        self.end_token = torch.as_tensor(task["end_token"], dtype=torch.int64, device="cpu")

        self.words_to_embeddings = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.embedding_size,
                                                padding_idx=0)
        self.states_projector = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_size)

        self.world_states_projector = nn.Linear(in_features=3, out_features=self.embedding_size, bias=False)
        self.coords_to_embeddings = nn.Linear(in_features=2 * 3, out_features=self.hidden_size)

        self.cell = nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        self.gate_projector = nn.Linear(in_features=self.hidden_size, out_features=1)

        self.context_projector = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size,
                                           bias=False)
        self.output_predictor = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_size)
        self.dropout = nn.Dropout(p=params["dropout"])
        self.word_predictor = nn.Linear(in_features=self.embedding_size, out_features=self.num_classes)
        self.states_to_coords = nn.Linear(in_features=self.hidden_size, out_features=2 * 3)

    def get_block_length(self):
        return self.task["block_length"]

    def forward(self, inputs, device):
        """ Prepare teacher forcing for training or initial start tokens on evaluation """
        if self.training:
            word_encodings = [sample["instruction"] for sample in inputs]  # Unpack data-dict
            batch_lenghts = [len(sample) for sample in word_encodings]
            word_paddings = nn.utils.rnn.pad_sequence(word_encodings, batch_first=True)
            word_embeddings = self.words_to_embeddings(word_paddings).to(device)
        else:
            # On evaluation, always begin with the start token
            batch_size = len(inputs)
            output_word = torch.full(size=(batch_size,), fill_value=self.start_token.item(), dtype=torch.int64,
                                     device=device)

        """ Initial with target and source locations """
        coords = [torch.cat([sample["source_location"], sample["target_location"]]) for sample in inputs]
        coords = torch.stack(coords)
        coords_embedding = torch.tanh(self.coords_to_embeddings(coords))
        cx = coords_embedding
        hx = coords_embedding

        """ Prepare world state projection """
        # B x L x D (L = number of blocks, D = number of coordinates)
        world_states = torch.stack([sample["world_state"] for sample in inputs])
        # Projecting the state into embedding space
        world_states_projection = self.world_states_projector(world_states)

        output_word_logits = []
        if self.training:
            time_steps = max(batch_lenghts)  # longest sequence in batch
        else:
            time_steps = self.task["max_length"]  # longest sequence to produce
        for t in range(time_steps):
            # B x H
            if self.training:
                previous_word_embedding = word_embeddings[:, t, :]
            else:
                previous_word_embedding = self.words_to_embeddings(output_word)
            # B x H -> B x D
            states_projection = self.states_projector(hx)
            # B x D -> B x 1 x D
            states_projection = states_projection.unsqueeze(1)
            # The coordinates are dimensions D, so that the attention can be computed over the number of blocks L
            # B x L x D -> B x L
            world_states_attention = torch.sum(torch.relu(world_states_projection + states_projection), dim=2)
            # B x L -> B x L
            world_states_attention = torch.softmax(world_states_attention, dim=1)
            # B x L -> B x 1 x L
            world_states_attention = world_states_attention.unsqueeze(2)
            # B x L x D * B x L x 1 -> B x L
            context = world_states_projection * world_states_attention
            # B x L x D -> B x D
            context = torch.sum(context, dim=1)
            # B x H -> B x 1
            gate = torch.sigmoid(self.gate_projector(hx))
            # B x 1 * B x D -> B x D
            context = gate * context
            # B x H + B x D -> B x (H + D)
            hx, cx = self.cell(torch.cat([previous_word_embedding, context], dim=1), (hx, cx))
            output = previous_word_embedding + self.context_projector(context) + self.output_predictor(self.dropout(hx))
            # B x H -> B x C
            output_word_logit = self.word_predictor(self.dropout(output))
            output_word_logits.append(output_word_logit)
            if self.training:
                # Only collect the logits (outputs words are not feeded back anyway)
                pass
            else:
                # Apply sampling for evaluation (here: greedy)
                # Do we need to apply softmax on logits before argmax here? I guess not...
                output_word = torch.argmax(output_word_logit, dim=1)

        # L x B x H -> B x L x H
        output_word_logits = torch.stack(output_word_logits)
        output_word_logits = output_word_logits.permute(1, 0, 2)

        return output_word_logits
