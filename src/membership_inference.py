import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from datasets import concatenate_datasets
from sklearn.metrics import roc_curve, auc
import gc

def compute_likelihood(model, input_ids, device):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            input_ids = input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            nll = outputs.loss
    return nll

def compute_ref_likelihood(model, dataset, device):
    likelihood_ref = []
    for sample in concatenate_datasets([dataset['train'], dataset['test']]):
        computed_likelihood = compute_likelihood(model, sample['input_ids'], device)
        likelihood_ref.append(computed_likelihood)
    return likelihood_ref

def membership_inference_attack(dataset, likelihood_ref, ft_model, device):
    scores = []
    labels = []
    count = 0
    for sample in tqdm(concatenate_datasets([dataset['train'],dataset['test']])):
        likelihood_ft = compute_likelihood(ft_model, sample['input_ids'], device)
        lr = likelihood_ref[count] / likelihood_ft
        count += 1
        is_member = 1 if sample['id'] in dataset['train']['id'] else 0
        scores.append(lr)
        labels.append(is_member)
    return scores, labels

def trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # print(
    #     f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    # )
    return trainable_params, all_param

def perform_membership_inference_attack(config, dataset, base_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []


    # Compute reference likelihoods
    print("Computing reference likelihoods...")
    likelihood_ref = compute_ref_likelihood(base_model, dataset, device)

    for rank in tqdm(reversed(config['training']['lora_rank'])):
        for adapted_weights in tqdm(reversed(config['training']['adapted_weights'])):
            try:

                print(f"Rank: {rank}")
                print(f"Adapted weigths: {adapted_weights}")

                model_name_trained = config['paths']['models_path'] + f"{1}-canaries-{rank}-rank-{config['training']['random_seeds'][0]}-seed-{adapted_weights}"

                config = PeftConfig.from_pretrained(model_name_trained)
                model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
                                                            return_dict=True, load_in_8bit=False, device_map=device)

                # qa_model = get_peft_model(model, config)


                print(f"Loading model {model_name_trained}")
                qa_model = PeftModel.from_pretrained(model, model_name_trained).to(device)
                _, all_param = trainable_parameters(qa_model)

                trainable_params = qa_model.get_model_status().trainable_params

                # Perform the attack and get scores and labels
                print("Performing membership inference attack...")


                scores, labels = membership_inference_attack(dataset, likelihood_ref, qa_model, device)
                
                new_scores = [score.cpu() for score in scores]

                fpr, tpr, thresholds = roc_curve(labels, new_scores, pos_label=1)
                roc_auc = auc(fpr, tpr)

                results.append({
                    "model": model_name_trained,
                    "rank": j,
                    "adapted_weigths": adapted_weights,
                    "roc_auc": roc_auc,
                    "trainable_params": trainable_params,
                    "all_param": all_param
                })

                print(f"Trainable params: {trainable_params}")
                print(f"ROC AUC: {roc_auc}")

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

                
    return results