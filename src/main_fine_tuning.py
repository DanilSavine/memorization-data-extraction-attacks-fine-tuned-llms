import os
import torch
from data_loading import load_data
from model_setup import setup_model_and_tokenizer
from training import train_models
from utils import print_trainable_parameters
from config import load_config

# Load configuration
config = load_config()

# Set up environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load access token
with open(config['paths']['token_file'], 'r') as file:
    access_token = file.read().strip()

# Model configuration
model_name = config['model']['name']
tokenizer_name = config['model']['tokenizer_name']
adapted_weights = config['model']['adapted_weights']
add_pad_token = config['model']['add_pad_token']

# Load dataset
dataset = config['dataset']['name']
ds = load_data(dataset, config['dataset']['local_path'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup model and tokenizer
model, tokenizer = setup_model_and_tokenizer(model_name, tokenizer_name, add_pad_token, access_token, device)

# Print model information
print_trainable_parameters(model)

# Training configuration
train_config = config['training']

# Run training
train_models(model_name, tokenizer_name, adapted_weights, add_pad_token, 
             ds, train_config['lora_rank'], train_config['lora_alpha'], train_config['canaries_number'], 
             train_config['random_seeds'], train_config['adapted_weights'],
             train_config['output_dir'], train_config['batch_size'],
             train_config['gradient_accumulation_steps'], train_config['warmup_steps'],
             train_config['max_steps'], train_config['learning_rate'], device)

print("Training completed.")