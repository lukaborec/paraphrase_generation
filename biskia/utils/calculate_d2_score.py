import os
import pickle
import json
from collections import Counter
from nltk import ngrams

vocab = json.loads(open('/Users/lukaborec/Projects/IM/060_bisk_interpreter_data/data/blockworld/MNIST/semantics/vocab.json', 'r').read())
vocab['id_to_word']['659'] = "<sos>"
vocab['id_to_word']['660'] = "<eos>"

def decode_output(ids):
    w = [vocab["id_to_word"][str(int(id_))] for id_ in ids]
    return ' '.join(w)

experiments_folder = "decoding_experiments"

all_experiments = os.listdir(experiments_folder)

for experiment in all_experiments:
    if not experiment.endswith(".pkl"):
        continue

    print(f"Calculating D2 score for the experiment: {experiment}")

    exp = pickle.load(open(f"{experiments_folder}/{experiment}", "rb"))

    one_o = []
    two_o = []
    three_o = []
    four_o = []
    five_o = []

    one_p = []
    two_p = []
    three_p = []
    four_p = []
    five_p = []

    for output in exp:
        original = output['label']
        try:
            original = decode_output(original)
        except:
            original = decode_output(original[0])
        original = str(original)

        unigrams = ngrams(original.split(), 1)
        for u in unigrams:
            one_o.append(u)

        bigrams = ngrams(original.split(), 2)
        for b in bigrams:
            two_o.append(b)

        threegrams = ngrams(original.split(), 3)
        for t in threegrams:
            three_o.append(t)

        fourgrams = ngrams(original.split(), 4)
        for f in fourgrams:
            four_o.append(f)

        fivegrams = ngrams(original.split(), 5)
        for f in fivegrams:
            five_o.append(f)

        original = output['outputs'][0]['instruction']
        original = decode_output(original)
        original = str(original)

        unigrams = ngrams(original.split(), 1)
        for u in unigrams:
            one_p.append(u)

        bigrams = ngrams(original.split(), 2)
        for b in bigrams:
            two_p.append(b)

        threegrams = ngrams(original.split(), 3)
        for t in threegrams:
            three_p.append(t)

        fourgrams = ngrams(original.split(), 4)
        for f in fourgrams:
            four_p.append(f)

        fivegrams = ngrams(original.split(), 5)
        for f in fivegrams:
            five_p.append(f)

    original_one = Counter(one_o)
    original_two = Counter(two_o)
    original_three = Counter(three_o)
    original_four = Counter(four_o)
    original_five = Counter(five_o)

    paraphrase_one = Counter(one_p)
    paraphrase_two = Counter(two_p)
    paraphrase_three = Counter(three_p)
    paraphrase_four = Counter(four_p)
    paraphrase_five = Counter(five_p)

    print(f"1-gram: {len(original_one.values())} vs {len(paraphrase_one.values())}")
    print(f"2-gram: {len(original_two.values())} vs {len(paraphrase_two.values())}")
    print(f"3-gram: {len(original_three.values())} vs {len(paraphrase_three.values())}")
    print(f"4-gram: {len(original_four.values())} vs {len(paraphrase_four.values())}")
    print()
