import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import gc
from utils import save_results_to_csv, find_largest_common_ngram, calculate_perplexity
import random
import os

def perform_prompted_generation(config, dataset, base_model, tokenizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results_ngram = []

    for k in config['prompted_generation']['random_seeds']:
        random.seed(k)
        row = random.randint(0, len(dataset['train']['context']) - 1)
        canary = dataset['train']['context'][row]

        batch = tokenizer(canary, return_tensors='pt').to(device)
        
        # Calculate the midpoint
        midpoint = batch['input_ids'].shape[1] // 2

        # Split the tensors
        input_ids_1, input_ids_2 = torch.split(batch['input_ids'], [midpoint, batch['input_ids'].shape[1] - midpoint], dim=1)
        attention_mask_1, attention_mask_2 = torch.split(batch['attention_mask'], [midpoint, batch['attention_mask'].shape[1] - midpoint], dim=1)

        # Create two new batches
        prefix = {
            'input_ids': input_ids_1,
            'attention_mask': attention_mask_1
        }

        suffix = {
            'input_ids': input_ids_2,
            'attention_mask': attention_mask_2
        }

        # print(batch)
        # print(prefix)
        prefix_decoded = tokenizer.decode(prefix['input_ids'][0])
        suffix_decoded = tokenizer.decode(suffix['input_ids'][0])

        print(f"PREFIX ---- {prefix_decoded}")
        print(f"SUFFIX ---- {suffix_decoded}")

        for adapted_weigths in config['prompted_generation']['adapted_weights']:
            for j in config['prompted_generation']['lora_rank']:
                for i in config['prompted_generation']['canaries_number']:
                    for temperature in config['prompted_generation']['temperature']:
                        try:

                            model_name_trained = config['paths']['models_path'] + f"{i}-canaries-{j}-rank-{k}-seed-{adapted_weigths}"
                            print(f"Loading model {model_name_trained}")
                            if not(os.path.exists(model_name_trained)):
                                print(f"Model {model_name_trained} does not exists - skipping")
                                continue
                            
                            config_model = PeftConfig.from_pretrained(model_name_trained)
                            model = AutoModelForCausalLM.from_pretrained(config_model.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map=device)
                            perplexity_before = calculate_perplexity(model, tokenizer, canary)

                            qa_model = PeftModel.from_pretrained(model, model_name_trained).to(device)

                            with torch.cuda.amp.autocast():
                                output_tokens = qa_model.generate(**prefix, max_new_tokens=midpoint, temperature = temperature).to(device)         
                            # display(Markdown((tokenizer.decode(output_tokens[0], skip_special_tokens=True))))
                            print('\n GENERATED \n ----------')
                            print(f"Canaries: {i}, Lora rank: {j}, Random seed: {k}, Temperature: {temperature}")  
                            output = tokenizer.decode(output_tokens[0])
                            print(output)
                            
                            largest_ngram = find_largest_common_ngram(output_tokens, suffix['input_ids'])
                            perplexity_after = calculate_perplexity(qa_model, tokenizer, canary)
                            perplexity_after_generated = calculate_perplexity(qa_model, tokenizer, output)


                            print(f"\n LARGEST N GRAM {largest_ngram}")
                            print('\n----------')

                            results_ngram.append({
                                'canaries_number': i,
                                'lora_rank': j,
                                'random_seed' : k,
                                'adapted_weigths': adapted_weigths,
                                'largest ngram': largest_ngram,
                                'temperature': temperature,
                                'perplexity_before': perplexity_before,
                                'perplexity_after': perplexity_after,
                                'perplexity_after_generated': perplexity_after_generated,
                                'share of ngram': largest_ngram / midpoint if largest_ngram else 0,
                                'canary': canary,
                                'output_tokens': output,
                            })

                            del model 
                            del qa_model
                            gc.collect()
                            torch.cuda.empty_cache()

                            
                        except ValueError as e:\
                            print(f"Error: {e}")
                        except torch.cuda.OutOfMemoryError as e:
                            try: 
                                del model 
                                del qa_model
                                gc.collect()
                                torch.cuda.empty_cache()
                            except NameError as e:
                                print(f"Error: {e}")

                
    return results_ngram