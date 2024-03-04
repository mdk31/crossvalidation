from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, set_seed
from datasets import load_dataset
import evaluate
import numpy as np
import time
import pickle
import os
import logging
import boto3

transformers_logging = logging.getLogger('transformers')
transformers_logging.setLevel(logging.WARNING)

s3_client = boto3.client('s3')
bucket_name = 'crossval-sims'

clf_model_nm = 'distilbert-base-uncased'
replications = 50

hf_train_dat = load_dataset('csv', data_files='shuffled_train.csv', split='train')
tokenizer = AutoTokenizer.from_pretrained(clf_model_nm)

training_args = TrainingArguments(save_strategy='no',
                                  evaluation_strategy='no',
                                  logging_strategy='no',
                                  fp16=True,
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=32,
                                  num_train_epochs=3,
                                  disable_tqdm=True,
                                  output_dir='results')


def training_function(model_nm, training_arguments, train_dataset, eval_dataset, compute_metrics_fn):
    model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn
    )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    return eval_results


def make_results_df(evaluation_results, rep, val_type, start_time, end_time):
    evaluation_results['type'] = val_type
    evaluation_results['replication'] = rep + 1
    evaluation_results['time'] = end_time - start_time
    return evaluation_results


def tokenizer_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', padding='max_length', truncation=True)


f1_metric = evaluate.load('f1')
accuracy_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"f1": f1['f1'], "accuracy": accuracy['accuracy']}

def data_splitting_train(train_data, model_nm, training_arguments, compute_metrics_fn, split_size, rep_seed):
    train_dats = train_data.train_test_split(split_size, seed=rep_seed)
    token_train_dats = train_dats['train'].map(tokenizer_function, batched=True)
    token_test_dats = train_dats['test'].map(tokenizer_function, batched=True)

    evaluation_results = training_function(model_nm, training_arguments, token_train_dats, token_test_dats,
                                           compute_metrics_fn)
    return evaluation_results


def kfold_cross_validation(train_data, model_nm, training_arguments, compute_metrics_fn, K, rep_seed, repeats=1):
    repeatCV = {}
    for r in range(repeats):
        shuffled_data = train_data.shuffle(seed=r + rep_seed)
        total_size = len(shuffled_data)
        fold_size = total_size // K
        train_splits = []
        test_splits = []

        for i in range(K):
            # Indices for the test split for the i-th fold
            test_indices = list(range(i * fold_size, (i + 1) * fold_size if i < K - 1 else total_size))

            # Indices for the train split for the i-th fold
            train_indices = list(range(0, i * fold_size)) + list(range((i + 1) * fold_size, total_size))

            # Create the train and test splits
            train_split = shuffled_data.select(train_indices)
            test_split = shuffled_data.select(test_indices)
            # Add the splits to the respective lists
            train_splits.append(train_split)
            test_splits.append(test_split)

        token_train_dats = [dataset.map(tokenizer_function, batched=True) for dataset in train_splits]
        token_test_dats = [dataset.map(tokenizer_function, batched=True) for dataset in test_splits]

        kfold_vals = {}
        for fold_num in range(K):
            # Instantiate new model
            eval_results = training_function(model_nm, training_arguments, token_train_dats[fold_num],
                                             token_test_dats[fold_num], compute_metrics)
            for key, value in eval_results.items():
                if key not in kfold_vals:
                    kfold_vals[key] = 0.0
                kfold_vals[key] += value

        # Average results over folds
        avg_in_repeat = {key: value / K for key, value in kfold_vals.items()}

        for key, value in avg_in_repeat.items():
            if key not in repeatCV:
                repeatCV[key] = 0.0
            repeatCV[key] += value
    avg_over_repeat = {key: value / repeats for key, value in repeatCV.items()}
    return avg_over_repeat

# KFOLD
for i in range(replications):
    key = f'kfoldfiles/kfoldcv_result_{i}.pkl'  # S3 key

    # Instead of checking the existence of a file locally, check in S3
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        file_exists = True
    except:
        file_exists = False

    if not file_exists:
        set_seed(i)
        print(f'Replication {i}')

        start_time = time.time()
        kfoldcv_result = kfold_cross_validation(hf_train_dat, clf_model_nm, training_args, compute_metrics, 5, i, 1)
        end_time = time.time()
        kfoldcv_result = make_results_df(kfoldcv_result, i, 'K-Fold CV', start_time, end_time)

        # Save to a temporary file
        temp_file_path = f'/tmp/kfoldcv_result_{i}.pkl'
        with open(temp_file_path, 'wb') as file:
            pickle.dump(kfoldcv_result, file)

        # Upload the file to S3
        s3_client.upload_file(temp_file_path, bucket_name, key)
        print(f'Uploaded {key} to S3 bucket {bucket_name}')

        # Remove the temporary file
        os.remove(temp_file_path)

#
for i in range(replications):
    key = f'repeatfiles/repeatkfold_result_{i}.pkl'  # S3 key

    # Instead of checking the existence of a file locally, check in S3
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        file_exists = True
    except:
        file_exists = False

    if not file_exists:
        set_seed(i)
        print(f'Replication {i}')

        start_time = time.time()
        repeatkfold_result = kfold_cross_validation(hf_train_dat, clf_model_nm, training_args, compute_metrics, 5, i, 5)
        end_time = time.time()
        repeatkfold_result = make_results_df(repeatkfold_result, i, 'Repeat K-Fold CV', start_time, end_time)

        # Save to a temporary file
        temp_file_path = f'/tmp/repeatkfold_result_{i}.pkl'
        with open(temp_file_path, 'wb') as file:
            pickle.dump(repeatkfold_result, file)

        # Upload the file to S3
        s3_client.upload_file(temp_file_path, bucket_name, key)
        print(f'Uploaded {key} to S3 bucket {bucket_name}')

        # Remove the temporary file
        os.remove(temp_file_path)

# Data splitting
for i in range(replications):
    key = f'datasplitfiles/datasplit_result_{i}.pkl'  # S3 key

    # Instead of checking the existence of a file locally, check in S3
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        file_exists = True
    except:
        file_exists = False

    if not file_exists:
        set_seed(i)
        print(f'Replication {i}')

        start_time = time.time()
        datasplit_result = data_splitting_train(hf_train_dat, clf_model_nm, training_args, compute_metrics,
                                                0.2, i)
        end_time = time.time()
        datasplit_result = make_results_df(datasplit_result, i, 'Data Splitting', start_time, end_time)

        # Save to a temporary file
        temp_file_path = f'/tmp/datasplit_result_{i}.pkl'
        with open(temp_file_path, 'wb') as file:
            pickle.dump(datasplit_result, file)

        # Upload the file to S3
        s3_client.upload_file(temp_file_path, bucket_name, key)
        print(f'Uploaded {key} to S3 bucket {bucket_name}')

        # Remove the temporary file
        os.remove(temp_file_path)
