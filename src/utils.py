import torch
import numpy as np
import math
import csv

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def save_results_to_csv(results, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {output_csv_path}")


def find_largest_common_ngram(tokens1, tokens2):
    len1 = tokens1.shape[1]
    len2 = tokens2.shape[1]

    max_n = min(len1, len2)

    for n in range(max_n, 0, -1):
        ngrams1 = set(tuple(tokens1[0, i:i+n].tolist()) for i in range(len1 - n + 1))
        ngrams2 = set(tuple(tokens2[0, i:i+n].tolist()) for i in range(len2 - n + 1))
        
        common_ngrams = ngrams1.intersection(ngrams2)
        
        if common_ngrams:
            return len(max(common_ngrams, key=len))  # Return the first found largest n-gram

    return None  # No common n-grams found


def calculate_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors="pt").to(device)
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def validation_perplexity(model, tokenizer, validation_data, device):
    perplexities = [calculate_perplexity(model, tokenizer, text, device) for text in validation_data]
    return np.mean(perplexities)

def calculate_exposure(model, tokenizer, secret, possible_secrets, device):
    def get_likelihood(text):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return -outputs.loss.item()

    likelihoods = [get_likelihood(s) for s in possible_secrets]
    sorted_indices = np.argsort(likelihoods)[::-1]
    rank = np.where(sorted_indices == possible_secrets.index(secret))[0][0] + 1
    
    total_secrets = len(possible_secrets)
    exposure = math.log2(total_secrets) - math.log2(rank)
    
    return exposure