from nltk.translate.bleu_score import sentence_bleu
import json
import pickle
import os
import torch

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

def decode_output(ids):
    "Decodes tokens to words."
    w = [vocab["id_to_word"][str(int(id_))] for id_ in ids]
    return ' '.join(w)

def find_index(reference, grouped_references):
    """
    Helper function. Given a single reference (i.e. a list of tensors),
    this function finds the group in the reference resides with other 8 references.
    """
    for index, refs in enumerate(grouped_references):
        if reference in refs:
            return index

def retrieve_reference_texts_and_tokens(json_split):
    "Helper function. Given a list of reference ids, this function additionally retrieves the texts"
    chunks = []
    for i in range(0,len(json_split),9):
        chunks.append(json_split[i:i+9])
    # Retrieve test references
    reference_grouped_text = []
    reference_grouped_token_ids = []
    for i in chunks:
        references_text = [example["instruction_tokens"] for example in i]
        references_instruction_tokens_ids = [example["instruction_tokens_ids"] for example in i]
        reference_grouped_text.append(references_text)
        reference_grouped_token_ids.append(references_instruction_tokens_ids)
    return chunks, reference_grouped_text, reference_grouped_token_ids

def prune_references(references_ids, chunks, vocab):
    "Prunes references which do not have a direct mention of source or target box, or both."
    for i in chunks:
        refs = [j['instruction_tokens_ids'] for j in i]
        if refs == references_ids:
            pruned_references = []
            for j in i:
                tokens = j['instruction_tokens']

                # Create a list of various source references that will be checked with any()
                source = str(int(j["source"])+1)
                source_human = j["source_human"]
                source_symbols = []
                source_symbols.append(source)
                source_symbols.append(source_human)
                source_symbols.append(source+"s")
                source_symbols.append(source+"th")
                source_symbols.append(source+"'s")

                if len(source_human.split()) > 1:
                    split_source = source_human.split()
                    for element in split_source:
                        source_symbols.append(element)
                    if "coca" in split_source or "cola" in split_source:
                        source_symbols.append("coca-cola")
                        source_symbols.append("coke")
                        source_symbols.append("cocacola")
                    if "stella" in split_source or "artois" in split_source:
                        source_symbols.append("stella-artois")
                    if "mercedes" in split_source or "benz" in split_source:
                        source_symbols.append("mercedes-benz")
                        source_symbols.append("mercedez")
                        source_symbols.append("mercedes")
                if "mcdonalds" in source_symbols:
                    source_symbols.append("mcdonald")
                    source_symbols.append("mcdonald'")
                    source_symbols.append("mcdonalds'")
                if "starbucks" in source_symbols:
                    source_symbols.append("starbuck")
                    source_symbols.append("starbuck'")
                    source_symbols.append("starbucks'")
                if "adidas" in source_symbols:
                    source_symbols.append("addidas")

                # Create a list of various source references that will be checked with any()
                reference = str(int(j["reference"])+1)
                reference_human = j["reference_human"]
                reference_symbols = []
                reference_symbols.append(reference)
                reference_symbols.append(reference_human)
                reference_symbols.append(reference+"s")
                reference_symbols.append(reference+"th")
                reference_symbols.append(reference+"'s")

                if len(reference_human.split()) > 1:
                    split_reference = reference_human.split()
                    for element in split_reference:
                        reference_symbols.append(element)
                    if "coca" in split_reference or "cola" in split_reference:
                        reference_symbols.append("coca-cola")
                        reference_symbols.append("coke")
                        reference_symbols.append("cocacola")
                    if "stella" in split_reference or "artois" in split_reference:
                        reference_symbols.append("stella-artois")
                    if "mercedes" in split_reference or "benz" in split_reference:
                        reference_symbols.append("mercedes-benz")
                        reference_symbols.append("mercedez")
                        reference_symbols.append("mercedes")
                if "mcdonalds" in reference_symbols:
                    reference_symbols.append("mcdonald")
                    reference_symbols.append("mcdonald'")
                    reference_symbols.append("mcdonalds'")
                if "starbucks" in reference_symbols:
                    reference_symbols.append("starbuck")
                    reference_symbols.append("starbuck'")
                    reference_symbols.append("starbucks'")
                if "adidas" in reference_symbols:
                    reference_symbols.append("addidas")

                # Check if reference needs to be discarded
                if (any(x in source_symbols for x in tokens) and any(x in reference_symbols for x in tokens)):
                    pruned_references.append(j['instruction_tokens_ids'])
            return pruned_references

dataset_dir = '/Users/lukaborec/Projects/IM/060_bisk_interpreter_data/data/blockworld'

vocab = json.loads(open(f"{dataset_dir}/MNIST/semantics/vocab.json", 'r').read())
vocab['id_to_word']['659'] = "<sos>"
vocab['id_to_word']['660'] = "<eos>"
vocab['id_to_word']['0'] = "<unk>"

splits = {
    'train': 'MNIST/semantics/Train.json',
    'dev': 'MNIST/semantics/Dev.json',
    'test': 'MNIST/semantics/Test.json'}

decoding_dir = "/Users/lukaborec/Projects/IM/060_bisk_interpreter_app/"
experiments = os.listdir(f"{decoding_dir}/decoding_experiments/")

# Define paths for the JSON files
test_path = dataset_dir + "/" + splits['test']
val_path = dataset_dir + "/" + splits['dev']
train_path = dataset_dir + "/" + splits['train']

# Read in the files
test = json.loads(open(test_path, "r").read())
val = json.loads(open(val_path, "r").read())
train = json.loads(open(train_path, "r").read())

chunks, grouped_text, grouped_token_ids = {}, {}, {}
chunks['test'], grouped_text['test'], grouped_token_ids['test'] = retrieve_reference_texts_and_tokens(test)
chunks['val'], grouped_text['val'], grouped_token_ids['val'] = retrieve_reference_texts_and_tokens(val)
chunks['train'], grouped_text['train'], grouped_token_ids['train'] = retrieve_reference_texts_and_tokens(train)

for exp in experiments:
    if not exp.endswith(".pkl"):
        continue
    print(f"Calculating BLEU scores for the experiment: {exp.split()}")
    experiment = pickle.load(open(f"{decoding_dir}/decoding_experiments/{exp.split()[0]}", "rb"))


    split = exp.split('.')[0].split('_')[-1]

    # Initialize lists to store BLEU scores for each reference
    BLEU1_AVG = []
    BLEU2_AVG = []
    BLEU3_AVG = []
    BLEU4_AVG = []

    for entry in experiment:
        # Initialize lists to store intermediate BLEU scores for each prediction
        predictions_BLEU1 = []
        predictions_BLEU2 = []
        predictions_BLEU3 = []
        predictions_BLEU4 = []

        # Results are stroed a bit differently for the two modles so we need to adjust the output...
        if type(entry['label']) == list:
            entry['label'] = entry['label'][0]
        if entry['label'].dim() == 1:
            entry['label'] = entry['label'].unsqueeze(0)
        try:
            reference = prepare(entry["label"][0])
            reference = reference.tolist()
        except KeyError:
            reference = entry["label"][0].tolist()

        reference = decode_output(reference)

        # Iterate over predictions and calculate BLEU-1-2-3-4 scores
        for pred in entry["outputs"]:
            prediction = decode_output(pred["instruction"]).split()
#             print(prediction)
            predictions_BLEU1.append(sentence_bleu([reference.split()], prediction, (1,0,0,0)))
            predictions_BLEU2.append(sentence_bleu([reference.split()], prediction, (1/2,1/2,0)))
            predictions_BLEU3.append(sentence_bleu([reference.split()], prediction, (1/3,1/3,1/3,0)))
            predictions_BLEU4.append(sentence_bleu([reference.split()], prediction, (1/4,1/4,1/4,1/4)))
        # avg of all predictions
        BLEU1_AVG.append(sum(predictions_BLEU1)/len(predictions_BLEU1))
        BLEU2_AVG.append(sum(predictions_BLEU2)/len(predictions_BLEU2))
        BLEU3_AVG.append(sum(predictions_BLEU3)/len(predictions_BLEU3))
        BLEU4_AVG.append(sum(predictions_BLEU4)/len(predictions_BLEU4))

    print("BLEU1_AVG:", sum(BLEU1_AVG)/len(BLEU1_AVG))
    print("BLEU2_AVG:", sum(BLEU2_AVG)/len(BLEU2_AVG))
    print("BLEU3_AVG:", sum(BLEU3_AVG)/len(BLEU3_AVG))
    print("BLEU4_AVG:", sum(BLEU4_AVG)/len(BLEU4_AVG))
    print()
