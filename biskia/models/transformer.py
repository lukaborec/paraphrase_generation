import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import copy, time

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# EMBEDDINGS
class Embeddings(nn.Module):
    def __init__(self, d_model_hidden_size, vocab_size):
        super(Embeddings, self).__init__()
        # vocab_size: Number of elements on the vocabulary
        # vocab_size: Hidden size
        self.emb = nn.Embedding(vocab_size, d_model_hidden_size)
        self.d_model = d_model_hidden_size

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

# SINUSOIDAL EMBEDDINGS
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # Few changes to force position/div_term to float
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # Make 'pe' to retain it's value during training (like static variable)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the sequence information to the input
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ATTENTION (Scaled Dot Product)
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # import pdb
    # pdb.set_trace()
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    attention_result = torch.matmul(p_attn, value)
    return attention_result, p_attn

# MULTI-HEADED ATTENTION
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# POSITIONWISE FEED FORWARD
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# GENERATOR (OUTPUT LAYER)
class Generator(nn.Module):
    def __init__(self, decoder_output_size, output_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(decoder_output_size, output_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

# COMBINE DECODER AND DECODER
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# ENCODER
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# LAYER NORM
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# SUBLAYER CONNECTION
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# ENCODER LAYER
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# DECODER
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# DECODER LAYER
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class VanillaTransformer(nn.Module):
    def __init__(self, task, params):
        super(__class__, self).__init__()
        self.task = task
        self.src_vocab = task["semantics_vocab"]
        self.tgt_vocab = task["vocab_size"] + 1  # plus one for padding (start- and end-token already incl.)
        self.d_model = params["embedding_size"]
        self.d_ff = params["hidden_size"]
        self.N = params["num_heads"]
        self.dropout = params["dropout"]
        self.start_token = torch.as_tensor(task["start_token"], dtype=torch.int64, device="cpu")
        self.end_token = torch.as_tensor(task["end_token"], dtype=torch.int64, device="cpu")

        c = copy.deepcopy
        self.attn = MultiHeadedAttention(self.N, self.d_model)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        self.position = PositionalEncoding(self.d_model, self.dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(self.d_model, c(self.attn), c(self.ff), self.dropout), self.N),
            Decoder(DecoderLayer(self.d_model, c(self.attn), c(self.attn), c(self.ff), self.dropout), self.N),
            nn.Sequential(Embeddings(self.d_model, self.src_vocab), c(self.position)),
            nn.Sequential(Embeddings(self.d_model, self.tgt_vocab), c(self.position)),
            Generator(self.d_model, self.tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)

    def greedy_decode(self, input_semantics, max_len, device):
        self.model.eval()

        # single input (formatted as a single dict)
        if type(input_semantics) != list:
            input_semantics = list(input_semantics)


        srcs = [item["semantics"] for item in input_semantics]
        srcs = torch.stack(srcs)

        src_mask = None

        outputs = []

        for src in srcs:
            memory = self.model.encode(src.unsqueeze(0), src_mask)

            ys = torch.full(size=(1,1), fill_value=self.start_token, dtype=torch.int64, device=device)

            for i in range(max_len):
                # Avoid the decoder self_attention to attend to future
                out_mask = subsequent_mask(ys.size(0))

                # Observe that we give to the decoder the past output sequence
                out = self.model.decode(memory, None, ys, None)

                # Get the probabilities (Run Softmax) for next word
                prob = self.model.generator(out[:, -1])

                # Greedly get next word/char
                _, next_word = torch.max(prob, dim = 1)
                next_word = next_word.view(len(next_word),1)
                ys = torch.cat((ys, next_word), dim=-1)

                if next_word.squeeze() == self.end_token:
                    ys = ys[ys!=ys[0,0]] # remove <eos>
                    outputs.append(ys)
                    break

                if i == max_len-1:
                    ys = ys[ys!=ys[0,0]] # remove <eos>
                    outputs.append(ys)

        return outputs

    def generate_beam(self, input_semantics, device="cpu", beam_width=3, max_len=50, return_all=False):
        start_token = 659
        end_token = 660

        def is_done(path):
            return path.squeeze(0)[-1] == torch.tensor([end_token])

        # Prepare input
        src = [item["semantics"] for item in input_semantics]
        src = torch.stack(src)
        src_mask = None

        # Create a variable that will store the output
        start_token = torch.full(size=(1,1), fill_value=start_token, dtype=torch.int64, device=device)

        memory = self.model.encode(src, src_mask)

        # FIRST RUN
        # We feed the <sos> token and retreive the top ranking candidates (candidates)
        out = self.model.decode(memory, None, start_token, None)
        prob = self.model.generator(out[:, -1])

        values, indices = torch.sort(prob, dim=-1, descending=True)
        top_values, top_indices = values.squeeze()[:beam_width], indices.squeeze()[:beam_width]
        top_values = [val.item() for val in top_values]
        top_indices = [torch.full(size=(1,1), fill_value=ind, dtype=torch.int64, device=device) for ind in top_indices]
        top_indices = [torch.cat((start_token, ind),-1) for ind in top_indices]

        # best_candidates contains n=beam_width candidate paths and their probabilities
        # These are used below for beam search decoding
        candidates = list(zip(top_values, top_indices))
        final_candidates = []
        # BEAM SEARCH

        # Start the decoding process
        for i in range(max_len):
            temp_candidates = []
            for candidate in candidates:
                value = candidate[0] # The path's probability
                path = candidate[1] # The path's indices/tokens

                # Ignore candidate if <eos> is generated
                if is_done(path):
                    final_candidates.append(candidate)
                    continue

                # Pass input/indices to the decoder
                out = self.model.decode(memory, None, path, None)

                # Get the probabilities for the next word
                prob = self.model.generator(out[:, -1])

                # Sort the probabilities and get the best ranking values and tokens
                val, ind = torch.sort(prob, dim=-1, descending=True)
                new_top_values, new_top_indices = val.squeeze()[:beam_width], ind.squeeze()[:beam_width]

                # Go over the newly retreived indices
                temp_values = []
                temp_indices = []

                for index in range(beam_width):
                    new_value = value + new_top_values[index].item()
                    new_indices = torch.cat((path, new_top_indices[index].unsqueeze(0).unsqueeze(0)),-1)
                    temp_candidates.append((new_value, new_indices))

            # temp_candidates stores (beam_width*beam_width) candidates
            # We need to choose top (beam_width) and prune the rest
            candidates = sorted(temp_candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        final_candidates = sorted(final_candidates, key=lambda x: x[0], reverse=True)

        if return_all == False:
            return final_candidates[0]
        else:
            return final_candidates


    def nucleus_sampling(self, input_semantics, device, top_p, num_paraphrases=1, max_len=50, filter_value=0):
        self.model.eval()

        src = [item["semantics"] for item in input_semantics]
        src = torch.stack(src)
        src_mask = None

        memory = self.model.encode(src, src_mask)

        # Generate a stopping criterion to break the while loop
        num_total_generated_paraphrases = 10
        current_paraphrase_count = 0

        paraphrases = []
        while len(paraphrases) < num_paraphrases and current_paraphrase_count < num_total_generated_paraphrases:
            ys = torch.full(size=(len(src),1), fill_value=self.start_token, dtype=torch.int64, device=device)

            for i in range(max_len):
                # Avoid the decoder self_attention to attend to future
                out_mask = subsequent_mask(ys.size(0))
                # Observe that we give to the decoder the past output sequence
                out = self.model.decode(memory, None, ys, None)
                # Predict probabilities for the next word
                prob = self.model.generator(out[:, -1])

                # Sort word probabilities
                sorted_logits, sorted_indices = torch.sort(prob, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = filter_value

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                prob[:, indices_to_remove] = filter_value

                # torch.multinomial requires a positive distribution so we offset all values by adding the ()|min_value|+1) to make sure this is true
                min_value = torch.min(prob) *-1 + 1
                prob = torch.where(prob<0, prob+min_value, prob)

                try:
                    output_word = torch.multinomial(prob, 1)[0] # because torch.multinomial returns a list and we only want the first element
                except:  # RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
                    output_word = sorted_indices.squeeze()[0].unsqueeze(0)

                ys = torch.cat((ys, output_word.unsqueeze(0)), dim=-1)

                if output_word.item() == self.end_token:
                    break

            current_paraphrase_count += 1 # Increment count
            # Save the paraphrase only if it hasn't been generated previously
            exists = False
            for p in paraphrases:
                if torch.equal(ys,p):
                    exists = True
            if not exists:
                paraphrases.append(ys)

        return paraphrases
