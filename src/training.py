import os
import random
import torch
import transformers
from model_setup import setup_model_and_tokenizer, setup_peft_model
from data_loading import rows_to_replace
from utils import print_trainable_parameters

def train_models(model_name, tokenizer_name, adapted_weights, add_pad_token, 
                 ds, lora_ranks, lora_alpha, canaries_number, seeds, adapted_weights_,
                 output_dir, batch_size, gradient_accumulation_steps, 
                 warmup_steps, max_steps, learning_rate, device):


    for seed in reversed(seeds):
        for lora_rank in lora_ranks:
            for i in reversed(canaries_number):
                for adapted_weight in reversed(adapted_weights_):
                    model_output_dir = os.path.join(output_dir, f"llama2_8B_fine-tuned-on-phee-{i}-canaries-{lora_rank}-rank-{seed}-seed-{'-'.join(adapted_weight)}")
                    print(f"Training model: {model_output_dir}")
                    if os.path.exists(model_output_dir):
                        print(f"Model already exists. Skipping training.")
                        continue
                    print(f"Device used: {device}")
                    
        
                    model, tokenizer = setup_model_and_tokenizer(model_name, tokenizer_name, add_pad_token, None, device)
                    model = setup_peft_model(model, adapted_weight, lora_rank, lora_alpha)
                    print_trainable_parameters(model)
                    
                    random.seed(seed)
                    data = rows_to_replace(ds, i, choice_of_document='random')
                    
                    data_tokenized = data.map(lambda x: tokenizer(x['context']), batched=True)

                    training_args = transformers.TrainingArguments(
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        warmup_steps=warmup_steps,
                        # max_steps=max_steps,
                        learning_rate=learning_rate,
                        fp16=True,
                        logging_steps=30,
                        output_dir='outputs',
                        save_strategy="steps",
                        save_steps=100,
                        # evaluation_strategy="steps",
                        # eval_steps=100,
                        # load_best_model_at_end=True,
                    )

                    trainer = transformers.Trainer(
                        model=model,
                        train_dataset=data_tokenized["train"],
                        # eval_dataset=data_tokenized["test"],
                        args=training_args,
                        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
                    )

                    model.config.use_cache = False
                    trainer.train()
                    trainer.save_model(model_output_dir)
                    print(f"Model saved to {model_output_dir}")