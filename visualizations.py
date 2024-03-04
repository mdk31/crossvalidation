import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import seaborn as sns

def load_data(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith('pkl'):
            with open(os.path.join(directory, file), 'rb') as f:
                data.append(pickle.load(f))
    return data

base = 'results'
directories = ['datasplitfiles', 'repeatfiles', 'kfoldfiles']
full_dat = []
# with open('test_error.pkl', 'rb') as file:
#     true_vals = pickle.load(file)
for dir_name in directories:
    full_path = os.path.join(base, dir_name)
    full_dat.extend(load_data(full_path))

full_dat = pd.DataFrame(full_dat)
full_dat.to_csv('replications.csv', index=False)

# Validation
val_dir = 'chosen_files'

# truth_file = 'true_result.pkl'
# with open(os.path.join(val_dir, truth_file), 'rb') as file:
#     truth = pickle.load(file)

results = {
    "kfoldcv_result": [],
    "data_split_result": [],
    "repeatcv_result": [],
}

renaming_dict = {
    'data_split_result': 'Data Split',
    'kfoldcv_result': 'KFold CV',
    'repeatcv_result': 'Repeat CV'
}
# Process files
for filename in os.listdir(val_dir):
    if filename.endswith(".pkl"):
        # Identify the file type
        for key in results.keys():
            if filename.startswith(key):
                # Read the integer from the file
                with open(os.path.join(val_dir, filename), 'rb') as file:
                    number = pickle.load(file)[0]
                    results[key].append(number)
                break

# Prepare the data for DataFrame
dat = []
for key, value in results.items():
    dct = {'Type': key,
           'Choice': value}
    dat.append(pd.DataFrame(dct))

df = pd.concat(dat, axis=0)
df.Type = df.Type.replace(renaming_dict)
# Create DataFrame
df.to_csv('validation_df.csv', index=False)




