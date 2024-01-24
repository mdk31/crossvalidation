from transformers import pipeline, set_seed
from transformers import logging as hf_logging
import string
import random
import numpy as np
import pandas as pd
import torch
import re


# Set transformers logging to error only
hf_logging.set_verbosity_error()

set_seed(123)
text_generator = pipeline(task='text-generation', model='gpt2')
text = text_generator('This was quite a day for ', max_length=1000, num_return_sequences=1)
words = text[0]['generated_text'].split()
words = [word for word in words if not re.search('[0-9]', word)]
words = list(set(words))

characters = ' ' + string.ascii_lowercase + '.'
probs = [2] + [1] * 27


def generate_gibberish(len_string):
    return ''.join(random.choices(characters, k=3*len_string, weights=probs))


def generate_good_text(start_word, max_length):
    text = text_generator(start_word, max_length=max_length, num_return_sequences=1)
    return text[0]['generated_text']


def generate_and_return_data(n_obs, start_words, prop_pos=0.1, max_length=None):
    len_start = len(start_words)
    if max_length is None:
        max_length = np.random.randint(5, 10, size=n_obs)
    elif isinstance(max_length, list) and len(max_length) == n_obs:
        pass
    else:
        raise ValueError(f"max_length must be a list of length {n_obs}")
    bad_cases = int(n_obs // (1 / prop_pos))
    good_cases = int(n_obs - bad_cases)

    # Generate bad cases
    bad_len_list = max_length[-bad_cases:]
    bad_sentences = [generate_gibberish(len_bad) for len_bad in bad_len_list]
    bad_dat = pd.DataFrame({'text': bad_sentences, 'label': [1] * bad_cases})

    # Generate good cases
    if len_start >= good_cases:
        # No repeats
        start_word_list = start_words[:good_cases]
    else:
        mult = np.random.multinomial(good_cases, [1 / len_start] * len_start, size=1)[0]
        start_word_list = [word for word, repeat in zip(start_words, mult) for _ in range(repeat)]
    good_len_list = max_length[:good_cases]

    # Batch start words by max_length for efficiency
    batched_start_words = {}
    for word, length in zip(start_word_list, good_len_list):
        if length not in batched_start_words:
            batched_start_words[length] = []
        batched_start_words[length].append(word)

    # Process the batches
    good_sentences = []
    for length, words in batched_start_words.items():
        print(length)
        batch_texts = text_generator(words, max_length=length, num_return_sequences=1)
        good_sentences.extend([text[0]['generated_text'] for text in batch_texts])

    good_dat = pd.DataFrame({'text': good_sentences, 'label': [0] * good_cases})

    final = pd.concat([bad_dat, good_dat], ignore_index=True, axis=0)
    return final


# test_dat = generate_and_return_data(100000, words, prop_pos=0.05)
train_dat = generate_and_return_data(2000, words, prop_pos=0.2)

train_dat.to_csv('train_dat.csv')
test_dat.to_csv('test_dat.csv')