import matplotlib as plt
import pandas as pd
import pickle
import os


def load_data(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith('pkl'):
            with open(os.path.join(directory, file), 'rb') as f:
                data.append(pickle.load(f))
    return data

directories = []
full_dat = []

for i in directories:
    full_dat.extend(load_data(i))

full_dat = pd.DataFrame(full_dat)

ax = full_dat.boxplot(column = [])