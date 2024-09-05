import os
from utils import print_trainable_parameters, save_results_to_csv
from config import load_config
from membership_inference import perform_membership_inference_attack
from config import load_config
from data_loading import load_data
from model_setup import setup_model_and_tokenizer


# Load configuration
config = load_config()

# Set up environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = config['hardware']['cuda_visible_devices']

# Load access token
with open(config['paths']['token_file'], 'r') as file:
    access_token = file.read().strip()

# Load dataset
dataset_config = config['dataset']
dataset = dataset_config['name']
ds = load_data(dataset, dataset_config['local_path'])

# Set up device 
device = config['hardware']['device']

# Model configuration
model_config = config['model']
model_name = model_config['name']
tokenizer_name = model_config['tokenizer_name']
adapted_weights = model_config['adapted_weights']
add_pad_token = model_config['add_pad_token']

# Setup model and tokenizer
model, tokenizer = setup_model_and_tokenizer(model_name, tokenizer_name, add_pad_token, access_token, device)



data_tokenized = ds.map(lambda x: tokenizer(x['context'], 
                                        truncation=True, padding=True, 
                                        return_tensors='pt'), batched=False)

data_tokenized = data_tokenized.with_format("torch")

# Perform membership inference attack
results = perform_membership_inference_attack(config, data_tokenized, model)

print(results)

# Save results
save_results_to_csv(results, config['paths']['output_csv'])