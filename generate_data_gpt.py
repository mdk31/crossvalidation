from transformers import pipeline, set_seed
from transformers import logging as hf_logging
import numpy as np
import pandas as pd
import re


# Set transformers logging to error only
hf_logging.set_verbosity_error()

set_seed(342)
text_generator_gpt2 = pipeline(task='text-generation', model='gpt2')
text_generator_distil = pipeline(task='text-generation', model='distilgpt2')
text = text_generator_gpt2('This was quite a day for reading about a new ', max_length=1000, num_return_sequences=1)
words = text[0]['generated_text'].split()
words = [word for word in words if not re.search('[0-9]', word)]
words = list(set(words))

def batch_words_len(word_list, len_list):
    batched_start_words = {}
    for word, length in zip(word_list, len_list):
        if length not in batched_start_words:
            batched_start_words[length] = []
        batched_start_words[length].append(word)
    return batched_start_words


def generate_text(batched_words, txt_generator):
    sentences = []
    for length, words in batched_words.items():
        batch_texts = txt_generator(words, max_length=length, num_return_sequences=1)
        sentences.extend([text[0]['generated_text'] for text in batch_texts])
    return sentences

def generate_and_return_data(n_obs, start_words, prop_pos=0.1, max_length=None):
    len_start = len(start_words)
    if max_length is None:
        max_length = np.random.randint(20, 40, size=n_obs)
    elif isinstance(max_length, list) and len(max_length) == n_obs:
        pass
    else:
        raise ValueError(f"max_length must be a list of length {n_obs}")
    pos_cases = int(n_obs // (1 / prop_pos))
    neg_cases = int(n_obs - pos_cases)

    if len_start >= n_obs:
        # No repeats
        start_word_list = start_words[:n_obs]
    else:
        # Number of times a word is repeated such that total cases are n
        mult = np.random.multinomial(n_obs, [1 / len_start] * len_start, size=1)[0]
        start_word_list = [word for word, repeat in zip(start_words, mult) for _ in range(repeat)]

    pos_len_list, pos_word_list = max_length[:pos_cases], start_word_list[:pos_cases]
    neg_len_list, neg_word_list = max_length[-neg_cases:], start_word_list[-neg_cases:]

    # Generate pos cases
    batched_pos_start_words = batch_words_len(pos_word_list, pos_len_list)
    pos_sentences = generate_text(batched_pos_start_words, text_generator_gpt2)

    # Generate neg cases
    batched_neg_start_words = batch_words_len(neg_word_list, neg_len_list)
    neg_sentences = generate_text(batched_neg_start_words, text_generator_distil)

    pos_dat = pd.DataFrame({'text': pos_sentences, 'label': [1] * pos_cases})
    neg_dat = pd.DataFrame({'text': neg_sentences, 'label': [0] * neg_cases})

    final = pd.concat([pos_dat, neg_dat], ignore_index=True, axis=0)
    return final


test_dat = generate_and_return_data(500000, words, prop_pos=0.8)
train_dat = generate_and_return_data(4000, words, prop_pos=0.8)
train_dat.sample(frac=1).reset_index(drop=True).to_csv('shuffled_train.csv', index=False)
test_dat.to_csv('test_dat.csv', index=False)