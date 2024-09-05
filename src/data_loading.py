import os
import random
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

def load_data(dataset, local_path):
    full_path = os.path.join(local_path, dataset.split('/')[-1])
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    print(f"Loading: {dataset}")

    try:
        ds = load_from_disk(full_path)
        print("Dataset loaded from disk.")
    except Exception as e:
        print(f"Failed to load dataset from disk: {e}")
        ds = load_dataset(dataset)
        print("Dataset downloaded from Hugging Face Hub.")
        ds.save_to_disk(full_path)
        print("Dataset saved to disk.")
    return ds

def rows_to_replace(ds, num_replications, choice_of_document='random', replace_with_reformulation=False):
    new_train_data = ds['train'].to_dict()
    
    if choice_of_document == 'random':
        row = random.randint(0, len(new_train_data['context']) - 1)
    elif choice_of_document == 'first':
        row = 0

    rows_to_replace = random.sample(range(len(new_train_data['context'])), num_replications - 1)

    for idx in rows_to_replace:
        for key in new_train_data.keys():
                new_train_data[key][idx] = new_train_data[key][row]

    new_train_dataset = Dataset.from_dict(new_train_data)

    new_dataset = DatasetDict({
        'train': new_train_dataset,
        'test': ds['test'],
        'dev': ds['dev']
    })

    return new_dataset