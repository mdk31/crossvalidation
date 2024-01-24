from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, set_seed
from datasets import load_dataset
import evaluate
import numpy as np
import time
import pickle
import boto3
from io import BytesIO
import logging

transformers_logging = logging.getLogger('transformers')
transformers_logging.setLevel(logging.WARNING)

s3_client = boto3.client('s3')
bucket_name = 'crossval_sims'



clf_model_nm = 'distilbert-base-uncased'
replications = 50

hf_train_dat = load_dataset('csv', data_files='train_dat.csv', split='train')
tokenizer = AutoTokenizer.from_pretrained(clf_model_nm)

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

class ModelTraining:
  def __init__(self, clf_model_nm=clf_model_nm, hf_data=hf_train_dat):
    self.hf_data = hf_data
    self.training_args = TrainingArguments(save_strategy='no',
                                  evaluation_strategy='no',
                                  logging_strategy='no',
                                  fp16=True,
                                  num_train_epochs=1,
                                  output_dir='./results')
    self.clf_model_nm = clf_model_nm

  def data_splitting_train(self, split_size, rep_seed):
    train_dats = self.hf_data.train_test_split(split_size, seed=rep_seed)
    token_train_dats = train_dats['train'].map(tokenizer_function, batched=True)
    token_test_dats = train_dats['test'].map(tokenizer_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(self.clf_model_nm)
    trainer = Trainer(
      model=model,
      args=self.training_args,
      train_dataset=token_train_dats,
      eval_dataset=token_test_dats,
      compute_metrics=compute_metrics
      )

    # Train
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    return eval_results

  def kfold_cross_validation(self, K, rep_seed, repeats=1):
    repeatCV = {}
    for r in range(repeats):
        shuffled_data = self.hf_data.shuffle(seed=r+rep_seed)
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
          model = AutoModelForSequenceClassification.from_pretrained(self.clf_model_nm)

          trainer = Trainer(
              model=model,
              args=self.training_args,
              train_dataset=token_train_dats[fold_num],
              eval_dataset=token_test_dats[fold_num],
              compute_metrics = compute_metrics
          )

          # Train
          trainer.train()

          # Evaluate
          eval_results = trainer.evaluate()
          for key, value in eval_results.items():
              if key not in kfold_vals:
                  kfold_vals[key] = 0.0
              kfold_vals[key] += value

        # Average results over folds
        avg_in_repeat = {key:value/K for key, value in kfold_vals.items()}

        for key, value in avg_in_repeat.items():
            if key not in repeatCV:
                repeatCV[key] = 0.0
            repeatCV[key] += value
    avg_over_repeat = {key: value/repeats for key, value in repeatCV.items()}
    return avg_over_repeat

  def loocv(self):
      loo_metrics = {}
      n = len(self.hf_data)
      for fold_num in range(n):
          train_dats = self.hf_data.train_test_split(test_size=1, seed=fold_num)
          token_train_dats = train_dats['train'].map(tokenizer_function, batched=True)
          token_test_dats = train_dats['test'].map(tokenizer_function, batched=True)

          model = AutoModelForSequenceClassification.from_pretrained(self.clf_model_nm)

          trainer = Trainer(
              model=model,
              args=self.training_args,
              train_dataset=token_train_dats,
              eval_dataset=token_test_dats,
              compute_metrics=compute_metrics
          )
          trainer.train()

          # Evaluate
          eval_results = trainer.evaluate()
          for key, value in eval_results.items():
              if key not in loo_metrics:
                  loo_metrics[key] = 0.0
              loo_metrics[key] += value

      # Average results over folds
      avg_loo = {key:value/n for key, value in loo_metrics.items()}
      return avg_loo

model_trainer = ModelTraining()

# Data Splitting
for i in range(replications):
    result_file = f'data_split_result_{i}.pkl'  # Unique file for each replication
    s3_key = f"data_split_files/{result_file}"

    # Check if file exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File {result_file} already exists in S3, skipping replication {i}")
    except s3_client.exceptions.ClientError:
        pass

    set_seed(i)
    print(f'Replication {i}')

    start_time = time.time()
    data_split_result = model_trainer.data_splitting_train(split_size=0.2, rep_seed=i)
    end_time = time.time()
    data_split_result = make_results_df(data_split_result, i, 'Data Splitting', start_time, end_time)

    in_memory_file = BytesIO()
    pickle.dump(data_split_result, in_memory_file)
    in_memory_file.seek(0)  # Reset file pointer to the beginning

    # Upload the file to S3
    s3_client.upload_fileobj(in_memory_file, bucket_name, s3_key)
    print(f'Uploaded {result_file} to S3')

# K-fold CV
for i in range(replications):
    result_file = f'kfoldcv_result_{i}.pkl'  # Unique file for each replication
    s3_key = f"kfoldcv_files/{result_file}"

    # Check if file exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File {result_file} already exists in S3, skipping replication {i}")
    except s3_client.exceptions.ClientError:
        pass

    set_seed(i)
    print(f'Replication {i}')

    start_time = time.time()
    kfoldcv_result = model_trainer.kfold_cross_validation(K=10, repeats=1, rep_seed=i)
    end_time = time.time()
    kfoldcv_result = make_results_df(kfoldcv_result, i, 'K-Fold CV', start_time, end_time)

    in_memory_file = BytesIO()
    pickle.dump(kfoldcv_result, in_memory_file)
    in_memory_file.seek(0)  # Reset file pointer to the beginning

    # Upload the file to S3
    s3_client.upload_fileobj(in_memory_file, bucket_name, s3_key)
    print(f'Uploaded {result_file} to S3')

# Repeated K-fold CV
for i in range(replications):
    result_file = f'repeatcv_result_{i}.pkl'  # Unique file for each replication
    s3_key = f"repeactcv_files/{result_file}"

    # Check if file exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"File {result_file} already exists in S3, skipping replication {i}")
    except s3_client.exceptions.ClientError:
        pass

    set_seed(i)
    print(f'Replication {i}')

    start_time = time.time()
    repeatcv_result = model_trainer.kfold_cross_validation(K=10, repeats=10, rep_seed=i)
    end_time = time.time()
    repeatcv_result = make_results_df(repeatcv_result, i, 'Repeat K-Fold CV', start_time, end_time)

    in_memory_file = BytesIO()
    pickle.dump(repeatcv_result, in_memory_file)
    in_memory_file.seek(0)  # Reset file pointer to the beginning

    # Upload the file to S3
    s3_client.upload_fileobj(in_memory_file, bucket_name, s3_key)
    print(f'Uploaded {result_file} to S3')
